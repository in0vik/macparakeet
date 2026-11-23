import AVFoundation
import os
import XCTest

@testable import MacParakeetCore

final class GeminiTranscribeEngineTests: XCTestCase {
    func testSilenceCutsPreserveOriginalFrameRanges() {
        let speech = Array(repeating: Float(0.1), count: 16_000)
        let quiet = Array(repeating: Float.zero, count: 48_000)
        XCTAssertEqual(
            GeminiTranscribeEngine.audibleRanges(in: speech + quiet + speech), [0..<20_000, 60_000..<80_000])
        XCTAssertEqual(GeminiTranscribeEngine.audibleRanges(in: speech + quiet), [0..<20_000])
        XCTAssertEqual(GeminiTranscribeEngine.audibleRanges(in: quiet + speech), [44_000..<64_000])
        XCTAssertTrue(GeminiTranscribeEngine.audibleRanges(in: quiet).isEmpty)
    }

    func testChunkBoundaryFallsWithinSilence() {
        let samples = Array(repeating: Float(0.1), count: 32_000) + Array(repeating: Float.zero, count: 8_000)
        XCTAssertEqual(GeminiTranscribeEngine.pauseCut(in: samples), 38_000)
        XCTAssertNil(GeminiTranscribeEngine.pauseCut(in: Array(repeating: 0.1, count: 48_000)))
    }

    func testWordTimingsUseOriginalPositionAndIgnoreOutOfRangeAnnotations() throws {
        let payload = Data(
            """
            {
              "status": "completed",
              "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Привет world", "annotations": [
                {"type": "word_info", "text": "Привет", "start_offset": "0.125s", "end_offset": "0.4s"},
                {"type": "word_info", "text": "world", "start_offset": 0.5, "end_offset": 0.75},
                {"type": "word_info", "text": "invalid", "start_offset": 0.9, "end_offset": 90}
              ]}]}]
            }
            """.utf8)
        let result = try GeminiTranscribeEngine.parseBatch(payload, status: 200, frameCount: 16_000)
        let words = GeminiTranscribeEngine.restoreTimeline(
            result.words, startFrame: 32_000, kept: [0..<6_400, 40_000..<49_600])
        XCTAssertEqual(result.text, "Привет world")
        XCTAssertEqual(result.engine, .gemini)
        XCTAssertEqual(words.map(\.startMs), [2125, 4600])
        XCTAssertEqual(words.map(\.endMs), [2400, 4850])
    }

    func testGeminiIsEligibleForLiveMeetingAndFinalRoutes() {
        let selection = SpeechEngineSelection(engine: .gemini, language: "ru")
        let capabilities = SpeechEngineCapabilityRegistry.capabilities(for: .gemini)
        XCTAssertNil(selection.language)
        XCTAssertTrue(capabilities.supportsNativeLiveDictation)
        XCTAssertEqual(
            MeetingSpeechPlan.resolve(live: selection, final: selection, liveCapabilities: capabilities).preview,
            selection)
    }

    func testLiveAndFileEndpointsWithOptInSample() async throws {
        guard let sample = ProcessInfo.processInfo.environment["MACPARAKEET_GEMINI_SMOKE_WAV"] else {
            throw XCTSkip("Set MACPARAKEET_GEMINI_SMOKE_WAV for the opt-in cloud smoke test")
        }
        let file = try AVAudioFile(
            forReading: URL(fileURLWithPath: sample), commonFormat: .pcmFormatFloat32, interleaved: false)
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        XCTAssertEqual(file.processingFormat.channelCount, 1)
        let engine = GeminiTranscribeEngine()
        let batch = try await engine.transcribe(
            audioPath: sample, job: .fileTranscription, onProgress: nil, wordTimestamps: true
        )
        XCTAssertFalse(batch.text.isEmpty)
        XCTAssertFalse(batch.words.isEmpty)
        try await engine.beginLiveDictation(language: nil, onPartial: { _ in })
        do {
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1600))
            while file.framePosition < file.length {
                try file.read(into: buffer)
                let samples = Array(
                    UnsafeBufferPointer(start: buffer.floatChannelData?[0], count: Int(buffer.frameLength)))
                try await engine.processLiveDictationSamples(samples)
                try await Task.sleep(for: .milliseconds(100))
            }
            let live = try await engine.finishLiveDictation()
            XCTAssertFalse(live.text.isEmpty)
            XCTAssertEqual(live.engineVariant, GeminiTranscribeEngine.liveModel)
        } catch {
            await engine.cancelLiveDictation()
            throw error
        }
        let meetingPreview = try await engine.transcribe(audioPath: sample, job: .meetingLiveChunk, onProgress: nil)
        XCTAssertFalse(meetingPreview.text.isEmpty)
        XCTAssertEqual(meetingPreview.engineVariant, GeminiTranscribeEngine.liveModel)
    }
}

extension GeminiTranscribeEngineTests {
    func testLiveDictationAtMicrophonePace() async throws {
        let file = try optInAudio("realtime-ru-en.wav")
        let recording = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).wav")
        try FileManager.default.copyItem(at: file.url, to: recording)
        defer { try? FileManager.default.removeItem(at: recording) }
        let repository = DictationRepository(dbQueue: try DatabaseManager().dbQueue)
        let scheduler = STTScheduler(runtime: STTRuntime(speechEngine: .gemini))
        let audio = MockAudioProcessor()
        await audio.configure(captureResult: recording)
        let service = DictationService(
            audioProcessor: audio, sttTranscriber: scheduler, dictationRepo: repository,
            shouldSaveAudio: { false }, shouldSaveDictationHistory: { true },
            shouldAttemptLiveDictationTranscription: { true })
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1600))
        let started = ContinuousClock.now
        try await service.startRecording()
        var firstPartial: Duration?
        while file.framePosition < file.length {
            try file.read(into: buffer)
            await audio.emitLiveSamples(Self.samples(in: buffer))
            try await Task.sleep(until: started + .seconds(Double(file.framePosition) / 16_000), clock: .continuous)
            if firstPartial == nil, !(await service.liveTranscript).isEmpty {
                firstPartial = started.duration(to: .now)
            }
        }
        let stopped = ContinuousClock.now
        let result = try await service.stopRecording()
        let saved = try XCTUnwrap(repository.fetch(id: result.dictation.id))
        let words = saved.rawTranscript.split(whereSeparator: \.isWhitespace).count
        let tail = saved.rawTranscript.localizedCaseInsensitiveContains("lighthouse")
        print(
            "GEMINI_DICTATION_SERVICE audio_seconds=\(Double(file.length) / 16_000) "
                + "first_partial=\(firstPartial ?? .zero) finish=\(stopped.duration(to: .now)) words=\(words) after=\(tail)"
        )
        XCTAssertNotNil(firstPartial)
        XCTAssertEqual(saved.engineVariant, GeminiTranscribeEngine.liveModel)
        XCTAssertGreaterThan(words, 65)
        XCTAssertTrue(tail)
    }

    func testLongLiveDictationAcrossGeminiRollover() async throws {
        let file = try optInAudio("long-ru-en.wav")
        let engine = GeminiTranscribeEngine()
        let started = ContinuousClock.now
        let updates = OSAllocatedUnfairLock(initialState: (first: Optional<Duration>.none, count: 0, text: ""))
        try await engine.beginLiveDictation(language: nil) { text in
            guard !text.isEmpty else { return }
            updates.withLock {
                $0.first = $0.first ?? started.duration(to: .now)
                $0.count += 1
                $0.text = text
            }
        }
        do {
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 1600))
            var rolloverDuration: Duration?
            while file.framePosition < file.length {
                let startFrame = file.framePosition
                try file.read(into: buffer)
                let sent = ContinuousClock.now
                try await engine.processLiveDictationSamples(Self.samples(in: buffer))
                if startFrame == 480 * 16_000 { rolloverDuration = sent.duration(to: .now) }
                try await Task.sleep(for: .milliseconds(25))
            }
            let stopped = ContinuousClock.now
            let final = try await engine.finishLiveDictation()
            let stopToResult = stopped.duration(to: .now)
            let snapshot = updates.withLock { $0 }
            let words = final.text.split(whereSeparator: \.isWhitespace).count
            let passes = final.text.localizedCaseInsensitiveContains("lighthouse")
            let prefix = final.text.localizedCaseInsensitiveContains("cobalt")
            let hasBothInPreview =
                snapshot.text.localizedCaseInsensitiveContains("cobalt")
                && snapshot.text.localizedCaseInsensitiveContains("lighthouse")
            print(
                "GEMINI_DICTATION audio_seconds=\(Double(file.length) / 16_000) "
                    + "first_partial=\(snapshot.first ?? .zero) updates=\(snapshot.count) "
                    + "wall=\(started.duration(to: .now)) finish=\(stopToResult) words=\(words) "
                    + "rollover=\(rolloverDuration ?? .zero) before=\(prefix) after=\(passes) preview_both=\(hasBothInPreview)"
            )
            XCTAssertEqual(final.engineVariant, GeminiTranscribeEngine.liveModel)
            XCTAssertNotNil(snapshot.first)
            XCTAssertGreaterThan(words, 600)
            XCTAssertTrue(passes, "Lost the final checkpoint")
            XCTAssertNotNil(rolloverDuration)
            XCTAssertTrue(prefix, "Lost the pre-rollover checkpoint")
            XCTAssertTrue(hasBothInPreview, "Live text did not retain the pre-rollover checkpoint")
        } catch {
            await engine.cancelLiveDictation()
            throw error
        }
    }

    func testMeetingPreviewAndFinalizationWithGemini() async throws {
        let file = try optInAudio("long-ru-en.wav")
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        let suite = "GeminiMeetingQA-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(true, forKey: SpeechEnginePreference.geminiWordTimestampsKey)
        let stt = STTClient(speechEngine: .gemini, defaults: defaults)
        let selection = SpeechEngineSelection(engine: .gemini)
        let preview = LiveChunkTranscriber(sttTranscriber: stt)
        let previewStarted = ContinuousClock.now
        let events = OSAllocatedUnfairLock(initialState: [(Duration, LiveChunkTranscriber.Event)]())
        let id = UUID()
        await preview.startSession(.init(id: id, chunkFolderURL: folder, speechEngine: selection)) { event in
            events.withLock { $0.append((previewStarted.duration(to: .now), event)) }
        }
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 80_000))
        let chunks: [(Int, AudioSource)] = [(0, .microphone), (4000, .microphone), (8000, .system)]
        for (startMs, source) in chunks {
            file.framePosition = Int64(startMs * 16)
            try file.read(into: buffer)
            await preview.enqueue(
                chunk: .init(samples: Self.samples(in: buffer), startMs: startMs, endMs: startMs + 5000),
                source: source)
        }
        let drained = await preview.waitForPendingTasksToDrain(timeout: .seconds(45))
        await preview.finishSession()
        XCTAssertTrue(drained)
        var microphones: [Int] = []
        var systems: [Int] = []
        for (elapsed, event) in events.withLock({ $0 }) {
            switch event {
            case .orderedResults(let results):
                for result in results {
                    XCTAssertEqual(result.result.engineVariant, GeminiTranscribeEngine.liveModel)
                    XCTAssertFalse(result.result.text.isEmpty)
                    switch result.source {
                    case .microphone: microphones.append(result.chunk.startMs)
                    case .system: systems.append(result.chunk.startMs)
                    }
                    print(
                        "GEMINI_MEETING_PREVIEW source=\(result.source.rawValue) start_ms=\(result.chunk.startMs) elapsed=\(elapsed) words=\(result.result.text.split(whereSeparator: \.isWhitespace).count)"
                    )
                }
            case .backpressureDrop: XCTFail("Dropped preview audio")
            case .transcriptionFailed(let message): XCTFail(message)
            }
        }
        XCTAssertEqual(microphones, [0, 4000])
        XCTAssertEqual(systems, [8000])

        let audioURL = folder.appendingPathComponent("meeting-microphone.wav")
        try FileManager.default.copyItem(at: file.url, to: audioURL)
        let recording = MeetingRecordingOutput(
            sessionID: id, displayName: "Gemini QA meeting", folderURL: folder, mixedAudioURL: audioURL,
            microphoneAudioURL: audioURL, systemAudioURL: folder.appendingPathComponent("system.wav"),
            durationSeconds: Double(file.length) / 16_000,
            sourceAlignment: .init(
                meetingOriginHostTime: nil,
                microphone: .init(
                    firstHostTime: nil, lastHostTime: nil, startOffsetMs: 0, writtenFrameCount: file.length,
                    sampleRate: 16_000),
                system: nil),
            speechEngine: selection, previewSpeechEngine: selection)
        let database = try DatabaseManager(path: folder.appendingPathComponent("qa.sqlite").path)
        let repository = TranscriptionRepository(dbQueue: database.dbQueue)
        let service = TranscriptionService(
            audioProcessor: AudioProcessor(), sttTranscriber: stt, transcriptionRepo: repository,
            shouldDiarizeMeetings: { false }, meetingArtifactStore: nil, meetingAutomationHookRunner: nil)
        let pending = try await service.prepareMeetingTranscription(recording: recording)
        let finalStarted = ContinuousClock.now
        let final = try await service.finalizeMeetingTranscription(recording: recording, updating: pending.id)
        let saved = try XCTUnwrap(repository.fetch(id: pending.id))
        let transcript = try XCTUnwrap(saved.rawTranscript)
        let words = transcript.split(whereSeparator: \.isWhitespace).count
        let before = transcript.localizedCaseInsensitiveContains("cobalt")
        let after = transcript.localizedCaseInsensitiveContains("lighthouse")
        let timings = try XCTUnwrap(saved.wordTimestamps)
        let lastWordEnd = timings.last?.endMs ?? 0
        print(
            "GEMINI_MEETING_FINAL audio_seconds=\(recording.durationSeconds) elapsed=\(finalStarted.duration(to: .now)) "
                + "words=\(words) timings=\(timings.count) last_word_ms=\(lastWordEnd) before=\(before) after=\(after)")
        XCTAssertEqual(final.id, pending.id)
        XCTAssertEqual(saved.sourceType, .meeting)
        XCTAssertEqual(saved.status, .completed)
        XCTAssertEqual(saved.engine, SpeechEnginePreference.gemini.rawValue)
        XCTAssertEqual(saved.engineVariant, GeminiTranscribeEngine.model)
        XCTAssertGreaterThan(words, 600)
        XCTAssertTrue(before)
        XCTAssertTrue(after)
        XCTAssertGreaterThan(lastWordEnd, 510_000)
        XCTAssertLessThanOrEqual(lastWordEnd, recording.playableDurationMs)
    }

    private func optInAudio(_ filename: String) throws -> AVAudioFile {
        guard let directory = ProcessInfo.processInfo.environment["MACPARAKEET_GEMINI_QA_DIR"] else {
            throw XCTSkip("Set MACPARAKEET_GEMINI_QA_DIR to the opt-in Gemini meeting and dictation fixtures")
        }
        let input = URL(fileURLWithPath: directory).appendingPathComponent(filename)
        let file = try AVAudioFile(forReading: input, commonFormat: .pcmFormatFloat32, interleaved: false)
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        XCTAssertEqual(file.processingFormat.channelCount, 1)
        return file
    }

    private static func samples(in buffer: AVAudioPCMBuffer) -> [Float] {
        Array(UnsafeBufferPointer(start: buffer.floatChannelData?[0], count: Int(buffer.frameLength)))
    }
}
