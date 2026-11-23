import Foundation
import XCTest

@testable import MacParakeetCore

final class ImportedRecordingCleanupTests: XCTestCase {
    func testDeletingAnImportKeepsOtherRecordingsAndTheOriginalFile() throws {
        let manager = FileManager.default
        let fixture = manager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let imports = fixture.appendingPathComponent("imported-recordings", isDirectory: true)
        let drop = imports.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try manager.createDirectory(at: drop, withIntermediateDirectories: true)
        defer { try? manager.removeItem(at: fixture) }

        let original = fixture.appendingPathComponent("Memo.m4a")
        let first = drop.appendingPathComponent("Memo.m4a")
        let second = drop.appendingPathComponent("Other Memo.m4a")
        try Data([1]).write(to: original)
        try manager.copyItem(at: original, to: first)
        try Data([2]).write(to: second)

        try TranscriptionAssetCleanup.removeImportedRecording(
            at: original, fileManager: manager, importsDirectory: imports)
        XCTAssertTrue(manager.fileExists(atPath: original.path))

        try TranscriptionAssetCleanup.removeImportedRecording(
            at: first, fileManager: manager, importsDirectory: imports)
        XCTAssertFalse(manager.fileExists(atPath: first.path))
        XCTAssertTrue(manager.fileExists(atPath: second.path))

        try TranscriptionAssetCleanup.removeImportedRecording(
            at: second, fileManager: manager, importsDirectory: imports)
        XCTAssertFalse(manager.fileExists(atPath: drop.path))
        XCTAssertTrue(manager.fileExists(atPath: original.path))
    }
}
