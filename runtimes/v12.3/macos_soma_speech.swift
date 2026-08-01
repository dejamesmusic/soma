import Foundation
import Speech
import AVFoundation
import Darwin

private func emit(_ kind: String, _ text: String = "") {
    let object: [String: String] = ["kind": kind, "text": text]
    guard let data = try? JSONSerialization.data(withJSONObject: object),
          let line = String(data: data, encoding: .utf8) else { return }
    FileHandle.standardOutput.write(Data((line + "\n").utf8))
    fflush(stdout)
}

private final class SpeechSession {
    private let localeIdentifier: String?
    private let engine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var recognizer: SFSpeechRecognizer?
    private var lastText = ""
    private var lastChange = Date()
    private var heardSpeech = false
    private var tapInstalled = false
    private var stopping = false
    private var finished = false
    private var timer: Timer?

    init(localeIdentifier: String?) {
        self.localeIdentifier = localeIdentifier
    }

    func start() {
        emit("status", "requesting local speech access")
        SFSpeechRecognizer.requestAuthorization { status in
            DispatchQueue.main.async {
                guard status == .authorized else {
                    self.fail("speech recognition permission was not granted")
                    return
                }
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    DispatchQueue.main.async {
                        guard granted else {
                            self.fail("microphone permission was not granted")
                            return
                        }
                        self.beginRecognition()
                    }
                }
            }
        }
    }

    private func beginRecognition() {
        let recognizer: SFSpeechRecognizer?
        if let localeIdentifier, !localeIdentifier.isEmpty,
           localeIdentifier.lowercased() != "auto" {
            recognizer = SFSpeechRecognizer(
                locale: Locale(identifier: localeIdentifier))
        } else {
            recognizer = SFSpeechRecognizer(locale: Locale.current)
        }
        guard let recognizer else {
            fail("speech recognition is unavailable for this language")
            return
        }
        guard recognizer.supportsOnDeviceRecognition else {
            fail("this mac cannot guarantee on-device speech recognition for this language")
            return
        }

        self.recognizer = recognizer
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true
        self.request = request

        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            fail("no microphone input is available")
            return
        }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) {
            [weak self] buffer, _ in
            self?.request?.append(buffer)
        }
        tapInstalled = true

        task = recognizer.recognitionTask(with: request) {
            [weak self] result, error in
            DispatchQueue.main.async {
                guard let self, !self.finished else { return }
                if let result {
                    let text = result.bestTranscription.formattedString
                    if text != self.lastText {
                        self.lastText = text
                        self.lastChange = Date()
                        self.heardSpeech = !text.trimmingCharacters(
                            in: .whitespacesAndNewlines).isEmpty
                        emit("partial", text)
                    }
                    if result.isFinal {
                        self.complete()
                        return
                    }
                }
                if let error, !self.stopping {
                    if self.heardSpeech {
                        self.complete()
                    } else {
                        self.fail(error.localizedDescription)
                    }
                }
            }
        }

        do {
            engine.prepare()
            try engine.start()
        } catch {
            fail("microphone could not start: \(error.localizedDescription)")
            return
        }

        emit("status", "listening locally")
        timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) {
            [weak self] _ in
            guard let self, !self.finished else { return }
            if self.heardSpeech && Date().timeIntervalSince(self.lastChange) > 1.4 {
                self.stopInput()
            }
        }

        DispatchQueue.global(qos: .userInitiated).async {
            _ = readLine()
            DispatchQueue.main.async { self.stopInput() }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 60) {
            self.stopInput()
        }
    }

    private func stopInput() {
        guard !finished, !stopping else { return }
        stopping = true
        if engine.isRunning { engine.stop() }
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        request?.endAudio()
        emit("status", "finishing transcript")
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            if !self.finished { self.complete() }
        }
    }

    private func cleanUp() {
        timer?.invalidate()
        timer = nil
        if engine.isRunning { engine.stop() }
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        request?.endAudio()
        task?.cancel()
        request = nil
        task = nil
    }

    private func complete() {
        guard !finished else { return }
        finished = true
        cleanUp()
        let text = lastText.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty {
            emit("error", "no speech was detected")
            Darwin.exit(2)
        }
        emit("final", text)
        Darwin.exit(0)
    }

    private func fail(_ message: String) {
        guard !finished else { return }
        finished = true
        cleanUp()
        emit("error", message)
        Darwin.exit(1)
    }
}

private let arguments = CommandLine.arguments
private let locale = arguments.count > 1 ? arguments[1] : nil
private let session = SpeechSession(localeIdentifier: locale)
session.start()
RunLoop.main.run()
