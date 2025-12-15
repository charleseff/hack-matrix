import SwiftUI
import SpriteKit

@main
struct HackApp: App {
    init() {
        // Check for --headless-cli (Python wrapper interface)
        if CommandLine.arguments.contains("--headless-cli") {
            HeadlessGameCLI().run()
            exit(0)
        }

        // Check for --headless-test command-line argument
        if CommandLine.arguments.contains("--headless-test") {
            DispatchQueue.global(qos: .userInitiated).async {
                HeadlessTest.runPerformanceTest()
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
    }
}

struct ContentView: View {
    let scene: SKScene = {
        let scene = GameScene()
        scene.size = CGSize(width: 600, height: 700)
        scene.scaleMode = .aspectFit
        return scene
    }()

    var body: some View {
        SpriteView(scene: scene)
            .frame(width: 600, height: 700)
            .background(Color.black)
    }
}
