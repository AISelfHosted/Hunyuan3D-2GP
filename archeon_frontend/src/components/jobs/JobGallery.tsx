import React, { useEffect, useState } from 'react';
import { apiClient, BASE_URL } from '../../api/client';
import { JobStatus } from '../../api/types';
import type { JobResponse } from '../../api/types';
import { RefreshCw, CheckCircle, Clock, XCircle, AlertTriangle, Download, Scissors } from 'lucide-react';

export const JobGallery: React.FC = () => {
    const [jobs, setJobs] = useState<JobResponse[]>([]);

    const fetchJobs = async () => {
        try {
            const res = await apiClient.get<JobResponse[]>('/jobs');
            // Sort by created_at desc
            setJobs(res.data.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
        } catch (err) {
            console.error(err);
        }
    };

    useEffect(() => {
        fetchJobs();
        const interval = setInterval(fetchJobs, 2000); // Polling every 2s
        return () => clearInterval(interval);
    }, []);

    const handleOptimize = async (e: React.MouseEvent, uid: string) => {
        e.stopPropagation();
        if (!confirm("Start decimation (50%)? This might take a few seconds.")) return;
        try {
            const res = await apiClient.post<{ file_path: string }>('/meshops/process', {
                job_uid: uid,
                action: 'decimate',
                ratio: 0.5
            });
            const fname = res.data.file_path.split('/').pop();
            if (confirm(`Optimization successful! Download ${fname}?`)) {
                window.open(`${BASE_URL}/files/${fname}`, '_blank');
            }
        } catch (err) {
            console.error(err);
            alert("Optimization failed. Check server logs.");
        }
    };

    const getStatusIcon = (status: string) => {
        switch (status) {
            case JobStatus.COMPLETED: return <CheckCircle size={16} className="text-green-500" />;
            case JobStatus.PROCESSING: return <RefreshCw size={16} className="text-blue-500 animate-spin" />;
            case JobStatus.FAILED: return <XCircle size={16} className="text-red-500" />;
            case JobStatus.QUEUED: return <Clock size={16} className="text-yellow-500" />;
            default: return <AlertTriangle size={16} className="text-gray-500" />;
        }
    };

    return (
        <div className="bg-archeon-panel border border-gray-700 rounded-lg p-6 h-full overflow-y-auto max-h-[600px]">
            <h3 className="text-lg font-semibold mb-4 text-gray-300 flex items-center justify-between sticky top-0 bg-archeon-panel py-2 z-10">
                <span>Recent Jobs</span>
                <button onClick={fetchJobs} className="text-gray-500 hover:text-white"><RefreshCw size={16} /></button>
            </h3>

            <div className="space-y-2">
                {jobs.length === 0 && <p className="text-gray-500 text-sm">No jobs found.</p>}

                {jobs.map(job => (
                    <div key={job.uid} className="bg-gray-800/50 p-3 rounded flex items-center justify-between hover:bg-gray-800 transition-colors cursor-pointer group">
                        <div className="flex items-center gap-3">
                            {getStatusIcon(job.status)}
                            <div className="text-sm">
                                <div className="font-mono text-gray-300 text-xs group-hover:text-archeon-primary transition-colors">{job.uid.slice(0, 8)}...</div>
                                <div className="text-xs text-gray-500">{new Date(job.created_at).toLocaleTimeString()}</div>
                            </div>
                        </div>
                        <div className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded ${job.status === JobStatus.COMPLETED ? 'bg-green-900/30 text-green-400' :
                            job.status === JobStatus.PROCESSING ? 'bg-blue-900/30 text-blue-400' :
                                job.status === JobStatus.FAILED ? 'bg-red-900/30 text-red-400' :
                                    'bg-gray-700 text-gray-400'
                            }`}>
                            {job.status}
                        </div>

                        {/* Actions */}
                        {job.status === JobStatus.COMPLETED && job.file_path && (
                            <div className="flex gap-1 ml-2">
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        const fname = job.file_path!.split('/').pop();
                                        window.open(`${BASE_URL}/files/${fname}`, '_blank');
                                    }}
                                    className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition-colors"
                                    title="Download Original"
                                >
                                    <Download size={14} />
                                </button>
                                <button
                                    onClick={(e) => handleOptimize(e, job.uid)}
                                    className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition-colors"
                                    title="Optimize (Decimate 50%)"
                                >
                                    <Scissors size={14} />
                                </button>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};
