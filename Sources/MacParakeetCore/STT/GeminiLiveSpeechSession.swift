import Foundation

actor GeminiLiveSpeechSession {
    private static let endpoint = URL(
        string:
            "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )!
    private let socket: URLSessionWebSocketTask
    private let onPartial: @Sendable (String) -> Void
    private var reader: Task<Void, Never>?
    private var ready = false
    private var failed = false
    private var closed = false
    private var sentAudio = false
    private var segments: [String] = []
    private var partial = ""
    private var lastChange = ContinuousClock.now
    private var pendingSamples: [Float] = []
    private static let frameSize = GeminiTranscribeEngine.sampleRate / 10

    private init(session: URLSession, key: String, onPartial: @escaping @Sendable (String) -> Void) {
        var request = URLRequest(url: Self.endpoint)
        request.setValue(key, forHTTPHeaderField: "x-goog-api-key")
        socket = session.webSocketTask(with: request)
        self.onPartial = onPartial
    }

    static func start(
        session: URLSession, key: String, onPartial: @escaping @Sendable (String) -> Void
    ) async throws -> GeminiLiveSpeechSession {
        try Task.checkCancellation()
        let live = GeminiLiveSpeechSession(session: session, key: key, onPartial: onPartial)
        await live.connect()
        return live
    }

    private func connect() {
        socket.resume()
        reader = Task { [weak self] in await self?.readLoop() }
    }

    private func readLoop() async {
        do {
            try await send([
                "setup": [
                    "model": "models/" + GeminiTranscribeEngine.liveModel,
                    "generationConfig": ["responseModalities": ["TEXT"]],
                    "inputAudioTranscription": ["languageCodes": [String]()],
                ]
            ])
            while !closed && !Task.isCancelled {
                let data: Data
                switch try await socket.receive() {
                case .data(let received): data = received
                case .string(let received): data = Data(received.utf8)
                @unknown default: throw GeminiSpeechError.invalidResponse
                }
                guard !closed, !Task.isCancelled else { return }
                let response = try JSONDecoder().decode(Response.self, from: data)
                if response.setupComplete != nil { ready = true }
                if response.error != nil { throw GeminiSpeechError.invalidResponse }
                guard let content = response.serverContent else { continue }
                if let interim = content.interimInputTranscription?.text, !interim.isEmpty {
                    partial = interim
                    lastChange = .now
                }
                if let final = content.inputTranscription?.text {
                    let text = final.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !text.isEmpty { segments.append(text) }
                    partial = ""
                    lastChange = .now
                }
                onPartial((segments + [partial]).filter { !$0.isEmpty }.joined(separator: " "))
            }
        } catch {
            if !closed { failed = true }
        }
    }

    private func awaitSetup() async throws {
        let deadline = ContinuousClock.now + .seconds(8)
        while !ready && !failed && !closed && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(25))
        }
        guard ready, !failed, !closed else { throw GeminiSpeechError.connectionFailed }
    }

    private func send(_ message: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: message)
        do {
            try await socket.send(.string(String(decoding: data, as: UTF8.self)))
        } catch {
            try Task.checkCancellation()
            throw GeminiSpeechError.connectionFailed
        }
    }

    func append(_ samples: [Float]) async throws {
        guard !samples.isEmpty else { return }
        pendingSamples.append(contentsOf: samples)
        while pendingSamples.count >= Self.frameSize {
            let frame = Array(pendingSamples.prefix(Self.frameSize))
            pendingSamples.removeFirst(Self.frameSize)
            try await sendAudio(frame)
        }
    }

    private func sendAudio(_ samples: [Float]) async throws {
        try await awaitSetup()
        try Task.checkCancellation()
        let data = GeminiTranscribeEngine.pcm(samples[...]).base64EncodedString()
        try await send(["realtimeInput": ["audio": ["data": data, "mimeType": "audio/pcm;rate=16000"]]])
        sentAudio = true
    }

    func finish() async throws -> STTResult {
        defer { close() }
        if !pendingSamples.isEmpty {
            let remaining = pendingSamples
            pendingSamples.removeAll()
            try await sendAudio(remaining)
        }
        if sentAudio {
            try await awaitSetup()
            try await send(["realtimeInput": ["audioStreamEnd": true]])
            let ended = ContinuousClock.now
            let deadline = ended + .seconds(4)
            while !failed && !closed && ContinuousClock.now < deadline {
                let now = ContinuousClock.now
                if !segments.isEmpty && partial.isEmpty && now >= ended + .milliseconds(750)
                    && now >= lastChange + .milliseconds(400)
                {
                    break
                }
                try await Task.sleep(for: .milliseconds(40))
            }
            guard !failed, !closed, partial.isEmpty, !segments.isEmpty else { throw GeminiSpeechError.liveIncomplete }
        }
        return STTResult(
            text: segments.joined(separator: " "), engine: .gemini, engineVariant: GeminiTranscribeEngine.liveModel
        )
    }

    func cancel() { close() }

    private func close() {
        closed = true
        reader?.cancel()
        reader = nil
        socket.cancel(with: .normalClosure, reason: nil)
    }

    private struct Response: Decodable {
        struct EmptyObject: Decodable {}
        struct Transcript: Decodable { let text: String? }
        struct Content: Decodable {
            let interimInputTranscription: Transcript?
            let inputTranscription: Transcript?
        }
        let setupComplete: EmptyObject?
        let serverContent: Content?
        let error: EmptyObject?
    }
}
