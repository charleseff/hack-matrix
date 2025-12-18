// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "HackMatrix",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "HackMatrix",
            targets: ["HackMatrix"]
        )
    ],
    targets: [
        .executableTarget(
            name: "HackMatrix",
            path: "HackMatrix",
            exclude: [
                "Info.plist",
                "Assets.xcassets"
            ],
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        )
    ]
)
