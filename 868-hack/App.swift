import SwiftUI
import SpriteKit

@main
struct HackApp: App {
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
