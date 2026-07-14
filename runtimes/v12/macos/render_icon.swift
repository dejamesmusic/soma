import AppKit

let outDir = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "macos/assets/soma.iconset"
try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

let specs: [(String, Int)] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

func render(path: String, px: Int) {
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: px,
        pixelsHigh: px,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    )!
    rep.size = NSSize(width: px, height: px)

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)

    NSColor(calibratedWhite: 0.0, alpha: 1.0).setFill()
    NSRect(x: 0, y: 0, width: px, height: px).fill()

    let fontSize = CGFloat(px) * 0.66
    let font = NSFont(name: "SFMono-Regular", size: fontSize)
        ?? NSFont(name: "Menlo", size: fontSize)
        ?? NSFont.monospacedSystemFont(ofSize: fontSize, weight: .regular)
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: NSColor(calibratedWhite: 0.94, alpha: 1.0)
    ]
    let text = "Φ" as NSString
    let size = text.size(withAttributes: attrs)
    let rect = NSRect(
        x: (CGFloat(px) - size.width) / 2,
        y: (CGFloat(px) - size.height) / 2 + CGFloat(px) * 0.035,
        width: size.width,
        height: size.height
    )
    text.draw(in: rect, withAttributes: attrs)

    NSGraphicsContext.restoreGraphicsState()

    if let data = rep.representation(using: .png, properties: [:]) {
        try? data.write(to: URL(fileURLWithPath: path))
    }
}

for (name, px) in specs {
    render(path: "\(outDir)/\(name)", px: px)
}
