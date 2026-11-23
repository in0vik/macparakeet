import AppKit
import MacParakeetCore
import SwiftUI
import UniformTypeIdentifiers

/// Voice Memos offers to export the recording when the drop is accepted.
@MainActor
struct PromisedRecordingDropTarget: NSViewRepresentable {
    @Binding var isTargeted: Bool
    var onFiles: ([URL]) -> Bool
    var onFailure: () -> Void

    func makeNSView(context: Context) -> TargetView {
        let view = TargetView()
        view.registerForDraggedTypes(PromisedRecordingDrop.types)
        return view
    }

    func updateNSView(_ view: TargetView, context: Context) {
        view.onTargeted = { isTargeted = $0 }
        view.onFiles = onFiles
        view.onFailure = onFailure
    }

    final class TargetView: NSView {
        var onTargeted: (Bool) -> Void = { _ in }
        var onFiles: ([URL]) -> Bool = { _ in false }
        var onFailure: () -> Void = {}

        override func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
            let pasteboard = sender.draggingPasteboard
            let accepted =
                PromisedRecordingDrop.canReceive(pasteboard) || !PromisedRecordingDrop.fileURLs(in: pasteboard).isEmpty
            onTargeted(accepted)
            return accepted ? .copy : []
        }

        override func draggingExited(_ sender: NSDraggingInfo?) { onTargeted(false) }

        override func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
            onTargeted(false)
            let pasteboard = sender.draggingPasteboard
            let urls = PromisedRecordingDrop.fileURLs(in: pasteboard)
            if !urls.isEmpty { return onFiles(urls) }
            return PromisedRecordingDrop.receive(pasteboard, onFiles: onFiles, onFailure: onFailure)
        }
    }
}

@MainActor
enum PromisedRecordingDrop {
    // Voice Memos offers its promises to destinations registered for file URLs.
    static let types =
        [.fileURL] + NSFilePromiseReceiver.readableDraggedTypes.map(NSPasteboard.PasteboardType.init(rawValue:))

    static func fileURLs(in pasteboard: NSPasteboard) -> [URL] {
        pasteboard.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? []
    }

    static func canReceive(_ pasteboard: NSPasteboard) -> Bool {
        !mediaPromises(in: pasteboard).isEmpty
    }

    static func receive(
        _ pasteboard: NSPasteboard, onFiles: @escaping ([URL]) -> Bool, onFailure: @escaping () -> Void
    ) -> Bool {
        let promises = mediaPromises(in: pasteboard)
        guard !promises.isEmpty else { return false }

        let destination = URL(fileURLWithPath: AppPaths.importedRecordingsDir, isDirectory: true)
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        } catch {
            onFailure()
            return false
        }

        let delivery = Delivery(
            destination: destination, count: promises.reduce(0) { $0 + max(1, $1.fileTypes.count) },
            onFiles: onFiles, onFailure: onFailure
        )
        delivery.start(promises)
        return true
    }

    private static func mediaPromises(in pasteboard: NSPasteboard) -> [NSFilePromiseReceiver] {
        let promises =
            pasteboard.readObjects(forClasses: [NSFilePromiseReceiver.self]) as? [NSFilePromiseReceiver] ?? []
        return promises.filter { promise in
            !promise.fileTypes.isEmpty
                && promise.fileTypes.allSatisfy { identifier in
                    guard let type = UTType(identifier) else { return false }
                    return type.conforms(to: .audio) || type.conforms(to: .movie)
                }
        }
    }

    @MainActor
    private final class Delivery {
        let destination: URL
        let onFiles: ([URL]) -> Bool
        let onFailure: () -> Void
        let queue = OperationQueue()
        var expected: Int
        var files: [URL] = []
        var failed = false
        var settled = false
        var deadline: Task<Void, Never>?

        init(destination: URL, count: Int, onFiles: @escaping ([URL]) -> Bool, onFailure: @escaping () -> Void) {
            self.destination = destination
            expected = count
            self.onFiles = onFiles
            self.onFailure = onFailure
        }

        func start(_ promises: [NSFilePromiseReceiver]) {
            queue.qualityOfService = .userInitiated
            deadline = Task {
                do { try await Task.sleep(for: .seconds(120)) } catch { return }
                failed = true
                finish()
            }
            for promise in promises {
                promise.receivePromisedFiles(atDestination: destination, operationQueue: queue) { url, error in
                    Task { @MainActor in self.received(url, succeeded: error == nil) }
                }
            }
        }

        private func received(_ url: URL, succeeded: Bool) {
            guard !settled else { return }
            let file = url.standardizedFileURL.resolvingSymlinksInPath()
            let parent = destination.standardizedFileURL.resolvingSymlinksInPath()
            if succeeded, file.deletingLastPathComponent().path == parent.path,
                AudioFileConverter.isSupported(extension: file.pathExtension),
                (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
            {
                files.append(file)
            } else {
                failed = true
            }
            expected -= 1
            if expected == 0 { finish() }
        }

        private func finish() {
            guard !settled else { return }
            settled = true
            deadline?.cancel()
            deadline = nil
            queue.cancelAllOperations()
            if !failed, !files.isEmpty, onFiles(files) { return }
            try? FileManager.default.removeItem(at: destination)
            onFailure()
        }
    }
}
