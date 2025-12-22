import Foundation

/// Logging levels
enum LogLevel: Int {
    case none = 0    // No logging
    case info = 1    // Important events only
    case debug = 2   // Verbose logging
}

/// Debug logging configuration
class DebugConfig {
    static var logLevel: LogLevel = .none
}

/// Info-level logging (important events only)
func infoLog(_ message: String) {
    guard DebugConfig.logLevel.rawValue >= LogLevel.info.rawValue else { return }
    fputs("INFO: \(message)\n", stderr)
    fflush(stderr)
}

func infoLog(_ category: String, _ message: String) {
    guard DebugConfig.logLevel.rawValue >= LogLevel.info.rawValue else { return }
    fputs("INFO [\(category)]: \(message)\n", stderr)
    fflush(stderr)
}

/// Debug-level logging (verbose, all details)
func debugLog(_ message: String) {
    guard DebugConfig.logLevel.rawValue >= LogLevel.debug.rawValue else { return }
    fputs("DEBUG: \(message)\n", stderr)
    fflush(stderr)  // Force flush to ensure logs appear immediately
}

func debugLog(_ category: String, _ message: String) {
    guard DebugConfig.logLevel.rawValue >= LogLevel.debug.rawValue else { return }
    fputs("DEBUG [\(category)]: \(message)\n", stderr)
    fflush(stderr)
}
