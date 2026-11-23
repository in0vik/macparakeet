import AVFoundation
import Foundation

/// Cloud STT is used only after the user chooses Gemini as a speech engine.
actor GeminiTranscribeEngine: NativeLiveDictating {
    static let model = "gemini-3.5-transcribe"
    static let liveModel = "gemini-3.5-transcribe-live"
    static let sampleRate = 16_000
    // 150 s of PCM16 WAV is about 4.8 MB (6.4 MB base64), comfortably
    // below the inline request limit and the 30-minute word timing limit.
    static let chunkSeconds = 150
    private static let interactionsURL = URL(string: "https://generativelanguage.googleapis.com/v1beta/interactions")!
    private let session: URLSession
    private let credential: @Sendable () throws -> String
    private struct LiveDictation {
        var connection: GeminiLiveSpeechSession
        let key: String
        let onPartial: @Sendable (String) -> Void
        var started = ContinuousClock.now
        var frames = 0
        var committed: [String] = []
    }
    private var liveDictation: LiveDictation?

    init(
        session: URLSession = GeminiTranscribeEngine.makeSession(),
        credential: @escaping @Sendable () throws -> String = { try GeminiTranscribeEngine.loadCredential() }
    ) {
        self.session = session
        self.credential = credential
    }

    nonisolated static func loadCredential() throws -> String {
        let environment = ProcessInfo.processInfo.environment
        if let key = nonEmpty(environment["GEMINI_PRIMARY_API_KEY"]) ?? nonEmpty(environment["GEMINI_API_KEY"]) {
            return key
        }
        do {
            if let key = nonEmpty(try LLMConfigStore().loadAPIKey(for: .gemini)) { return key }
        } catch {
            throw GeminiSpeechError.credentialUnavailable
        }
        throw GeminiSpeechError.missingCredential
    }

    private nonisolated static func nonEmpty(_ value: String?) -> String? {
        guard let key = value?.trimmingCharacters(in: .whitespacesAndNewlines), !key.isEmpty else { return nil }
        return key
    }

    nonisolated static func makeSession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 65
        configuration.timeoutIntervalForResource = 9 * 60
        return URLSession(configuration: configuration, delegate: GeminiSpeechNoRedirects(), delegateQueue: nil)
    }

    func prepare() throws { _ = try credential() }

    // The key is resolved at the start of each request/session so rotating a
    // saved credential never requires rebuilding the runtime.
    func isReady() -> Bool { (try? credential()) != nil }

    func transcribe(
        audioPath: String,
        job: STTJobKind,
        onProgress: (@Sendable (Int, Int) -> Void)?,
        wordTimestamps: Bool = false
    ) async throws -> STTResult {
        let key = try credential()
        let inputURL = URL(fileURLWithPath: audioPath)
        var convertedURL: URL?
        defer {
            if let convertedURL { try? FileManager.default.removeItem(at: convertedURL) }
        }
        let file: AVAudioFile
        if let original = try? Self.openAudio(inputURL), Self.isNormalized(original.processingFormat) {
            file = original
        } else {
            let temporary = try await AudioFileConverter().convert(fileURL: inputURL)
            convertedURL = temporary
            file = try Self.openAudio(temporary)
        }
        guard Self.isNormalized(file.processingFormat) else { throw GeminiSpeechError.unsupportedAudio }
        if job == .meetingLiveChunk { return try await transcribeMeetingChunk(file, key: key) }

        let total = file.length
        let capacity = Self.sampleRate * Self.chunkSeconds
        var texts: [String] = []
        var words: [TimestampedWord] = []
        while file.framePosition < total {
            try Task.checkCancellation()
            let start = file.framePosition
            var samples = try Self.read(file, limit: capacity)
            if samples.isEmpty { throw GeminiSpeechError.unsupportedAudio }
            if file.framePosition < total, let cut = Self.pauseCut(in: samples) {
                file.framePosition = start + Int64(cut)
                samples.removeLast(samples.count - cut)
            }
            let kept = Self.audibleRanges(in: samples)
            if !kept.isEmpty {
                // Keep one request per window even when it contains many pauses.
                let upload = kept.flatMap { samples[$0] }
                let result = try await transcribeSamples(upload[...], key: key, wordTimestamps: wordTimestamps)
                if !result.text.isEmpty { texts.append(result.text) }
                words.append(contentsOf: Self.restoreTimeline(result.words, startFrame: start, kept: kept))
            }
            onProgress?(Int(file.framePosition), Int(total))
        }
        return STTResult(text: texts.joined(separator: "\n"), words: words, engine: .gemini, engineVariant: Self.model)
    }

    private func transcribeMeetingChunk(_ file: AVAudioFile, key: String) async throws -> STTResult {
        // Preview jobs carry short capture chunks with absolute offsets.
        // The existing meeting pipeline owns overlap and deduplication.
        let samples = try Self.read(file, limit: Self.sampleRate * Self.chunkSeconds)
        guard file.framePosition >= file.length else { throw GeminiSpeechError.unsupportedAudio }
        guard !Self.audibleRanges(in: samples).isEmpty else {
            return STTResult(text: "", engine: .gemini, engineVariant: Self.liveModel)
        }
        let live = try await GeminiLiveSpeechSession.start(session: session, key: key, onPartial: { _ in })
        do {
            // The chunk is already in memory. Pace queued input at up to 4x
            // realtime so the preview can catch up without bursting the socket.
            for offset in stride(from: 0, to: samples.count, by: Self.sampleRate / 10) {
                try await live.append(Array(samples[offset..<min(offset + Self.sampleRate / 10, samples.count)]))
                try await Task.sleep(for: .milliseconds(25))
            }
            return try await live.finish()
        } catch {
            await live.cancel()
            throw error
        }
    }

    func beginLiveDictation(language: String?, onPartial: @escaping @Sendable (String) -> Void) async throws {
        guard liveDictation == nil else { throw STTError.engineBusy }
        let key = try credential()
        let connection = try await GeminiLiveSpeechSession.start(session: session, key: key, onPartial: onPartial)
        liveDictation = LiveDictation(connection: connection, key: key, onPartial: onPartial)
    }

    func processLiveDictationSamples(_ samples: [Float]) async throws {
        guard var state = liveDictation else { throw STTLiveDictationTranscriptionError.sessionNotActive }
        let sessionSeconds = 8 * 60
        if state.frames > 0
            && (state.frames + samples.count > sessionSeconds * Self.sampleRate
                || .now > state.started + .seconds(sessionSeconds))
        {
            let previous = state.connection
            let finished = try await previous.finish()
            guard liveDictation?.connection === previous else { throw CancellationError() }
            if !finished.text.isEmpty { state.committed.append(finished.text) }
            let prefix = state.committed.joined(separator: " ")
            let callback = state.onPartial
            let next = try await GeminiLiveSpeechSession.start(session: session, key: state.key) { partial in
                callback([prefix, partial].filter { !$0.isEmpty }.joined(separator: " "))
            }
            guard liveDictation?.connection === previous else {
                await next.cancel()
                throw CancellationError()
            }
            state.connection = next
            state.started = .now
            state.frames = 0
        }
        state.frames += samples.count
        liveDictation = state
        try await state.connection.append(samples)
    }

    func finishLiveDictation() async throws -> STTResult {
        guard let state = liveDictation else { throw STTLiveDictationTranscriptionError.sessionNotActive }
        defer {
            if liveDictation?.connection === state.connection { liveDictation = nil }
        }
        let final = try await state.connection.finish()
        guard liveDictation?.connection === state.connection else { throw CancellationError() }
        return STTResult(
            text: (state.committed + [final.text]).filter { !$0.isEmpty }.joined(separator: " "),
            engine: .gemini, engineVariant: Self.liveModel
        )
    }

    func cancelLiveDictation() async {
        let pending = liveDictation?.connection
        liveDictation = nil
        await pending?.cancel()
    }

    private func transcribeSamples(
        _ samples: ArraySlice<Float>, key: String, wordTimestamps: Bool
    ) async throws -> STTResult {
        let input = GeminiBatchRequest(
            input: [.init(data: Self.wav(samples).base64EncodedString())],
            generation_config: .init(
                transcription_config: .init(
                    mode: .init(
                        timestamp_granularities: wordTimestamps ? ["word"] : nil
                    )))
        )
        var request = URLRequest(url: Self.interactionsURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 65
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(key, forHTTPHeaderField: "x-goog-api-key")
        request.httpBody = try JSONEncoder().encode(input)

        let (firstData, firstResponse) = try await send(request)
        if firstResponse.statusCode == 429 {
            let delay = Double(firstResponse.value(forHTTPHeaderField: "Retry-After") ?? "") ?? 1
            guard delay.isFinite, delay <= 5 else { throw GeminiSpeechError.httpStatus(429) }
            try await Task.sleep(for: .seconds(max(0.5, delay)))
            let (data, response) = try await send(request)
            return try Self.parseBatch(data, status: response.statusCode, frameCount: samples.count)
        }
        return try Self.parseBatch(firstData, status: firstResponse.statusCode, frameCount: samples.count)
    }

    private func send(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { throw GeminiSpeechError.invalidResponse }
            return (data, http)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as GeminiSpeechError {
            throw error
        } catch {
            try Task.checkCancellation()
            throw GeminiSpeechError.connectionFailed
        }
    }

    nonisolated static func parseBatch(_ data: Data, status: Int, frameCount: Int) throws -> STTResult {
        guard (200..<300).contains(status) else { throw GeminiSpeechError.httpStatus(status) }
        guard let response = try? JSONDecoder().decode(GeminiBatchResponse.self, from: data),
            response.status == nil || response.status == "completed"
        else { throw GeminiSpeechError.invalidResponse }
        let content =
            response.steps?.filter { $0.type == "model_output" }.flatMap { $0.content ?? [] }
            ?? response.outputs ?? []
        let transcript = content.filter { $0.type == "text" || $0.type == "transcript" }
        let text = transcript.compactMap(\.text).joined(separator: "\n")
        let duration = Double(frameCount) / Double(sampleRate)
        let words = transcript.flatMap { $0.annotations ?? [] }.compactMap { item -> TimestampedWord? in
            guard item.type == "word_info", let word = item.text, !word.isEmpty,
                let start = item.startOffset?.value, let stop = item.endOffset?.value,
                start >= 0, stop >= start, start <= duration + 0.1, stop <= duration + 0.1
            else { return nil }
            return TimestampedWord(
                word: word, startMs: Int((min(duration, start) * 1000).rounded()),
                endMs: Int((min(duration, stop) * 1000).rounded()), confidence: 1
            )
        }
        guard !content.isEmpty || response.outputText != nil else { throw GeminiSpeechError.invalidResponse }
        return STTResult(
            text: (text.isEmpty ? response.outputText ?? "" : text).trimmingCharacters(in: .whitespacesAndNewlines),
            words: words, engine: .gemini, engineVariant: model
        )
    }

    nonisolated static func restoreTimeline(
        _ words: [TimestampedWord], startFrame: Int64, kept: [Range<Int>]
    ) -> [TimestampedWord] {
        guard !words.isEmpty else { return [] }
        var cursor = 0
        let timeline = kept.map { original in
            let begin = cursor
            cursor += original.count
            return (uploaded: begin..<cursor, originalStart: original.lowerBound)
        }
        let framesPerMillisecond = sampleRate / 1000
        return words.compactMap { word in
            let start = word.startMs * framesPerMillisecond
            let end = word.endMs * framesPerMillisecond
            guard let interval = timeline.first(where: { $0.uploaded.contains(start) }) else { return nil }
            let origin = startFrame + Int64(interval.originalStart - interval.uploaded.lowerBound)
            return TimestampedWord(
                word: word.word,
                startMs: Int((Double(origin + Int64(start)) / Double(framesPerMillisecond)).rounded()),
                endMs: Int(
                    (Double(origin + Int64(min(end, interval.uploaded.upperBound))) / Double(framesPerMillisecond))
                        .rounded()),
                confidence: word.confidence
            )
        }
    }

    private nonisolated static func openAudio(_ url: URL) throws -> AVAudioFile {
        try AVAudioFile(forReading: url, commonFormat: .pcmFormatFloat32, interleaved: false)
    }

    private nonisolated static func isNormalized(_ format: AVAudioFormat) -> Bool {
        format.sampleRate == Double(sampleRate) && format.channelCount == 1
    }

    private nonisolated static func read(_ file: AVAudioFile, limit: Int) throws -> [Float] {
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(limit)
            )
        else { throw GeminiSpeechError.unsupportedAudio }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else { throw GeminiSpeechError.unsupportedAudio }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    private nonisolated static func isQuiet(_ samples: ArraySlice<Float>) -> Bool {
        // Conservative -54 dBFS peak gate. Only clearly quiet spans are cut.
        samples.allSatisfy { abs($0) <= 0.002 }
    }

    nonisolated static func pauseCut(in samples: [Float]) -> Int? {
        let window = sampleRate / 4
        guard samples.count >= sampleRate * 2 else { return nil }
        for end in stride(from: samples.count, through: max(window, samples.count - sampleRate * 8), by: -window) {
            if isQuiet(samples[(end - window)..<end]) { return end - window / 2 }
        }
        return nil
    }

    nonisolated static func audibleRanges(in samples: [Float]) -> [Range<Int>] {
        guard !samples.isEmpty, !isQuiet(samples[...]) else { return [] }
        let window = sampleRate / 10
        let minimumQuiet = sampleRate * 2
        let pad = sampleRate / 4
        var ranges: [Range<Int>] = []
        var keptStart = 0
        var quietStart: Int?
        for index in stride(from: 0, to: samples.count, by: window) {
            let end = min(index + window, samples.count)
            if index < end, isQuiet(samples[index..<end]) {
                if quietStart == nil { quietStart = index }
                continue
            }
            if let quiet = quietStart, index - quiet >= minimumQuiet {
                if quiet > keptStart { ranges.append(keptStart..<(quiet + pad)) }
                keptStart = index - pad
            }
            quietStart = nil
        }
        if let quiet = quietStart, samples.count - quiet >= minimumQuiet {
            if quiet > keptStart { ranges.append(keptStart..<(quiet + pad)) }
        } else if keptStart < samples.count {
            ranges.append(keptStart..<samples.count)
        }
        return ranges
    }

    nonisolated static func pcm(_ samples: ArraySlice<Float>) -> Data {
        let values = samples.map { sample -> Int16 in
            let normalized = sample.isFinite ? max(-1, min(1, sample)) : 0
            return Int16(normalized * Float(Int16.max)).littleEndian
        }
        return values.withUnsafeBytes { Data($0) }
    }

    nonisolated static func wav(_ samples: ArraySlice<Float>) -> Data {
        let audio = pcm(samples)
        var result = Data("RIFF".utf8)
        func append<T: FixedWidthInteger>(_ value: T) {
            var littleEndian = value.littleEndian
            withUnsafeBytes(of: &littleEndian) { result.append(contentsOf: $0) }
        }
        append(UInt32(36 + audio.count))
        result.append(contentsOf: "WAVEfmt ".utf8)
        append(UInt32(16)); append(UInt16(1)); append(UInt16(1))
        append(UInt32(sampleRate)); append(UInt32(sampleRate * 2)); append(UInt16(2)); append(UInt16(16))
        result.append(contentsOf: "data".utf8)
        append(UInt32(audio.count))
        result.append(audio)
        return result
    }
}

private final class GeminiSpeechNoRedirects: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        completionHandler(nil)
    }
}

enum GeminiSpeechError: LocalizedError {
    case missingCredential, credentialUnavailable, unsupportedAudio, invalidResponse, connectionFailed, liveIncomplete
    case httpStatus(Int)

    var errorDescription: String? {
        switch self {
        case .missingCredential: "Add a Google Gemini API key in Settings → AI Provider to use Gemini speech."
        case .credentialUnavailable:
            "MacParakeet could not read the saved Google Gemini key. Check access in AI settings."
        case .unsupportedAudio: "Gemini speech could not read the audio as 16 kHz mono."
        case .invalidResponse: "Google Gemini returned an incomplete transcription response."
        case .connectionFailed: "The connection to Google Gemini failed."
        case .liveIncomplete: "Google Gemini Live did not finalize the captured speech."
        case .httpStatus(let code): "Google Gemini transcription returned HTTP \(code)."
        }
    }
}

private struct GeminiBatchRequest: Encodable {
    struct Input: Encodable {
        let type = "audio"
        let mime_type = "audio/wav"
        let data: String
    }
    struct Configuration: Encodable {
        struct Transcription: Encodable {
            struct Mode: Encodable {
                let type = "verbatim"
                let timestamp_granularities: [String]?
            }
            let language_codes: [String] = []
            let mode: Mode
        }
        let transcription_config: Transcription
    }
    let model = "models/" + GeminiTranscribeEngine.model
    let input: [Input]
    let generation_config: Configuration
    let store = false
}

private struct GeminiBatchResponse: Decodable {
    struct Step: Decodable {
        let type: String?
        let content: [Content]?
    }
    struct Content: Decodable {
        let type: String?
        let text: String?
        let annotations: [Annotation]?
    }
    struct Annotation: Decodable {
        let type: String?
        let text: String?
        let startOffset: Seconds?
        let endOffset: Seconds?
        enum CodingKeys: String, CodingKey {
            case type, text, startOffset = "start_offset", endOffset = "end_offset"
        }
    }
    struct Seconds: Decodable {
        let value: Double?
        init(from decoder: Decoder) throws {
            let input = try decoder.singleValueContainer()
            if let seconds = try? input.decode(Double.self) {
                value = seconds.isFinite ? seconds : nil
            } else if let duration = try? input.decode(String.self) {
                let numeric = Double(duration.hasSuffix("s") ? String(duration.dropLast()) : duration)
                value = numeric?.isFinite == true ? numeric : nil
            } else {
                value = nil
            }
        }
    }
    let status: String?
    let steps: [Step]?
    let outputs: [Content]?
    let outputText: String?
    enum CodingKeys: String, CodingKey { case status, steps, outputs, outputText = "output_text" }
}
