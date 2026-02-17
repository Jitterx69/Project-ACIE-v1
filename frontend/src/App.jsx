import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Activity, Server, Zap, Database, RefreshCw } from 'lucide-react';
import AuditLogTable from './components/AuditLogTable';
import LatentSpaceVisualizer from './components/LatentSpaceVisualizer';

const Card = ({ title, value, icon: Icon, subtext }) => (
    <div className="bg-surface border border-gray-800 rounded-xl p-6 shadow-lg">
        <div className="flex items-center justify-between mb-4">
            <h3 className="text-gray-400 text-sm font-medium">{title}</h3>
            <Icon className="w-5 h-5 text-gray-500" />
        </div>
        <div className="text-3xl font-bold text-white mb-1">{value}</div>
        <div className="text-xs text-gray-500">{subtext}</div>
    </div>
);

function App() {
    const [metrics, setMetrics] = useState({
        latency: [],
        throughput: 0,
        gpu: { util: 0, memory: 0 },
        requests: 0
    });

    // Fetch live metrics from backend
    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const response = await fetch('/api/dashboard/stats');
                if (!response.ok) throw new Error('Network response was not ok');

                const data = await response.json();

                // Transform backend data to frontend format
                setMetrics({
                    latency: data.latency_history.map(item => ({
                        time: item.time,
                        value: item.value
                    })),
                    throughput: Math.floor(data.total_requests / 60), // Approx req/min -> req/s placeholder or use real rate
                    gpu: {
                        util: data.gpu.length > 0 ? data.gpu[0].utilization : 0,
                        memory: data.gpu.length > 0 ? data.gpu[0].memory_used / (1024 * 1024) : 0 // Bytes to MB
                    },
                    requests: data.total_requests
                });
            } catch (error) {
                console.error("Failed to fetch metrics:", error);
            }
        };

        // Initial fetch
        fetchMetrics();

        // Poll every 2 seconds
        const interval = setInterval(fetchMetrics, 2000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="min-h-screen bg-background text-white p-8">
            <header className="mb-10 flex items-center justify-between">
                <div>
                    <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600">
                        ACI Engine
                    </h1>
                    <p className="text-gray-400 mt-2">Real-time Inference Monitoring & Control</p>
                </div>
                <div className="flex gap-4">
                    <button className="flex items-center gap-2 px-4 py-2 bg-surface border border-gray-700 rounded-lg hover:border-gray-500 transition-colors">
                        <RefreshCw className="w-4 h-4" />
                        Refresh
                    </button>
                    <div className="flex items-center gap-2 px-4 py-2 bg-surface border border-green-900/50 text-green-400 rounded-lg">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        System Online
                    </div>
                </div>
            </header>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <Card
                    title="Throughput"
                    value={`${metrics.throughput} req/s`}
                    icon={Activity}
                    subtext="Total processed: 1.2M"
                />
                <Card
                    title="Avg Latency"
                    value={`${metrics.latency.length > 0 ? metrics.latency[metrics.latency.length - 1].value.toFixed(1) : 0} ms`}
                    icon={Zap}
                    subtext="P95: 24.5ms"
                />
                <Card
                    title="GPU Utilization"
                    value={`${metrics.gpu.util}%`}
                    icon={Server}
                    subtext="NVIDIA A100-80GB"
                />
                <Card
                    title="Memory Usage"
                    value={`${(metrics.gpu.memory / 1024).toFixed(1)} GB`}
                    icon={Database}
                    subtext="Limit: 80 GB"
                />
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 bg-surface border border-gray-800 rounded-xl p-6 shadow-lg">
                    <h3 className="text-lg font-semibold mb-6">Inference Latency (Real-time)</h3>
                    <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={metrics.latency}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="time" stroke="#666" fontSize={12} tickCount={5} />
                                <YAxis stroke="#666" fontSize={12} unit="ms" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#18181b', borderColor: '#333' }}
                                    itemStyle={{ color: '#fff' }}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="value"
                                    stroke="#8884d8"
                                    strokeWidth={2}
                                    dot={false}
                                    activeDot={{ r: 4 }}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-surface border border-gray-800 rounded-xl p-6 shadow-lg">
                    <h3 className="text-lg font-semibold mb-6">Model Distribution</h3>
                    <div className="h-[300px] flex items-center justify-center text-gray-500">
                        Pie Chart Placeholder
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                <AuditLogTable />
                <LatentSpaceVisualizer />
            </div>
        </div>
    );
}

export default App;
