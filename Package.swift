// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "868-hack",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "868-hack",
            targets: ["868-hack"]
        )
    ],
    targets: [
        .executableTarget(
            name: "868-hack",
            path: "868-hack",
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
