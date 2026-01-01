import SwiftUI
import SpriteKit

@main
struct HackApp: App {
    init() {
        fputs("==> HackApp.init() called\n", stderr)
        fflush(stderr)

        // Check for --headless-cli (Python wrapper interface)
        if CommandLine.arguments.contains("--headless-cli") {
            fputs("==> Starting headless CLI mode\n", stderr)
            fflush(stderr)
            HeadlessGameCLI().run()
            exit(0)
        }

        // Check for --visual-cli mode
        if CommandLine.arguments.contains("--visual-cli") {
            fputs("==> Starting visual CLI mode (GUI with stdin/stdout)\n", stderr)
            fflush(stderr)
        }

        // Check for --headless-test command-line argument
        if CommandLine.arguments.contains("--headless-test") {
            DispatchQueue.global(qos: .userInitiated).async {
                HeadlessTest.runPerformanceTest()
            }
        }

        // Check for --run-tests command-line argument
        if CommandLine.arguments.contains("--run-tests") {
            GameLogicTests.runAllTests()
            exit(0)
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
