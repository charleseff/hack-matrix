// SPM entry point - headless CLI only
// For GUI app, use Xcode build which creates proper .app bundle

import Foundation
import HackMatrixCore

// Run headless CLI (works on both macOS and Linux)
let cli = HeadlessGameCLI()
cli.run()
