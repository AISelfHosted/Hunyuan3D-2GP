import React, { useEffect, useState } from 'react';
import type { SystemMetrics } from '../../api/types';
import { apiClient } from '../../api/client';
import { Server, Cpu, CircuitBoard, Activity } from 'lucide-react';

export const SystemMonitor: React.FC = () => {
    const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const response = await apiClient.get<SystemMetrics>('/system/metrics');
                setMetrics(response.data);
                setError(null);
            } catch (err) {
                console.error("Failed to fetch metrics", err);
                setError("Offline");
            }
        };

        fetchMetrics();
        const interval = setInterval(fetchMetrics, 2000);
        return () => clearInterval(interval);
    }, []);

    if (error) return <div className="text-red-500 text-sm flex items-center gap-2"><Activity size={16} /> Backend Offline</div>;
    if (!metrics) return <div className="text-gray-500 text-sm animate-pulse">Scanning system...</div>;

    return (
        <div className="bg-archeon-panel p-4 rounded-lg border border-gray-700 space-y-4 shadow-lg">
            <h3 className="text-sm font-semibold text-gray-400 flex items-center gap-2 border-b border-gray-700 pb-2">
                <Server size={16} className="text-archeon-primary" /> SYSTEM VITALS
            </h3>

            {/* CPU */}
            <div className="space-y-1">
                <div className="flex justify-between text-xs text-gray-300">
                    <span className="flex items-center gap-1"><Cpu size={12} /> CPU</span>
                    <span>{metrics.cpu_percent.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-blue-500 transition-all duration-500 ease-out"
                        style={{ width: `${metrics.cpu_percent}%` }}
                    />
                </div>
            </div>

            {/* RAM */}
            <div className="space-y-1">
                <div className="flex justify-between text-xs text-gray-300">
                    <span className="flex items-center gap-1"><CircuitBoard size={12} /> RAM</span>
                    <span>{metrics.memory.percent.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                    <div
                        className="h-full bg-purple-500 transition-all duration-500 ease-out"
                        style={{ width: `${metrics.memory.percent}%` }}
                    />
                </div>
                <div className="text-[10px] text-gray-500 text-right">
                    {(metrics.memory.used / 1024 / 1024 / 1024).toFixed(1)} GB / {(metrics.memory.total / 1024 / 1024 / 1024).toFixed(1)} GB
                </div>
            </div>

            {/* GPU */}
            {metrics.gpu && (
                <div className="space-y-1 pt-2 border-t border-gray-700/50">
                    <div className="flex justify-between text-xs text-gray-300">
                        <span className="flex items-center gap-1"><Activity size={12} /> VRAM ({metrics.gpu.name})</span>
                        <span>{((metrics.gpu.used_memory / metrics.gpu.total_memory) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                        <div
                            className="h-full bg-green-500 transition-all duration-500 ease-out"
                            style={{ width: `${(metrics.gpu.used_memory / metrics.gpu.total_memory) * 100}%` }}
                        />
                    </div>
                    <div className="text-[10px] text-gray-500 text-right">
                        {(metrics.gpu.used_memory / 1024).toFixed(1)} GB / {(metrics.gpu.total_memory / 1024).toFixed(1)} GB
                    </div>
                </div>
            )}
        </div>
    );
};
