import { SystemMonitor } from './components/monitoring/SystemMonitor';
import { CreateJobForm } from './components/jobs/CreateJobForm';
import { JobGallery } from './components/jobs/JobGallery';

function App() {
  return (
    <div className="min-h-screen bg-archeon-bg text-white p-8 font-sans">
      <header className="mb-8 flex justify-between items-center border-b border-gray-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
            Archeon 3D
          </h1>
          <p className="text-gray-400 text-sm">High-Fidelity Local Generation Engine</p>
        </div>
        <div className="text-xs text-gray-500 font-mono">
          v1.0.0-alpha (Phase 2 Preview)
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Sidebar / Monitor Area */}
        <div className="lg:col-span-1 space-y-6">
          <SystemMonitor />

          <div className="bg-archeon-panel p-4 rounded-lg border border-gray-700 shadow-lg">
            <h3 className="text-sm font-semibold text-gray-400 mb-2">QUICK ACTIONS</h3>
            <button className="w-full bg-archeon-primary/80 hover:bg-archeon-primary text-white py-2 rounded text-sm font-medium transition-colors mb-2">
              New Project
            </button>
            <button className="w-full bg-gray-700/50 hover:bg-gray-700 text-white py-2 rounded text-sm font-medium transition-colors">
              Documentation
            </button>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-3 space-y-8">
          <div className="bg-archeon-panel border border-gray-700 rounded-lg p-0 overflow-hidden shadow-xl">
            <CreateJobForm />
          </div>

          <JobGallery />
        </div>
      </div>
    </div>
  );
}

export default App;
