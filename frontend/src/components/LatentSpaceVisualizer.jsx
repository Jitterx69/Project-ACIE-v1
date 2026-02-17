import React, { useState, useEffect } from 'react';
import { fetchAnalysisPlots } from '../api/rService';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Network } from 'lucide-react';

// Mock data for scatter plot since we don't have real images yet
const mockData = [
    { x: 10, y: 30, z: 200 },
    { x: 30, y: 200, z: 260 },
    { x: 45, y: 100, z: 400 },
    { x: 50, y: 400, z: 280 },
    { x: 70, y: 150, z: 100 },
    { x: 100, y: 250, z: 500 },
];

const LatentSpaceVisualizer = () => {
    const [plots, setPlots] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadPlots = async () => {
            const data = await fetchAnalysisPlots();
            setPlots(data);
            setLoading(false);
        };
        loadPlots();
    }, []);

    if (loading) {
        return (
            <div className="bg-surface border border-gray-800 rounded-xl p-6 shadow-lg flex justify-center items-center h-[300px]">
                <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
        );
    }

    return (
        <div className="bg-surface border border-gray-800 rounded-xl p-6 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
                <Network className="w-5 h-5 text-purple-400" />
                Latent Space Visualization
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-black/30 p-4 rounded-lg border border-gray-800">
                    <h4 className="text-sm font-medium text-gray-400 mb-2">PC1 vs PC2 (Real-time Projection)</h4>
                    <div className="h-[200px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis type="number" dataKey="x" name="PC1" stroke="#666" fontSize={10} />
                                <YAxis type="number" dataKey="y" name="PC2" stroke="#666" fontSize={10} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#18181b', borderColor: '#333' }} />
                                <Scatter name="Vectors" data={mockData} fill="#8884d8" />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-2">
                    {plots.slice(0, 4).map((plot) => (
                        <div key={plot.id} className="relative group overflow-hidden rounded-lg border border-gray-800">
                            <img
                                src={plot.url}
                                alt={plot.title}
                                className="w-full h-full object-cover opacity-70 group-hover:opacity-100 transition-opacity"
                            />
                            <div className="absolute bottom-0 left-0 right-0 bg-black/60 p-1 text-[10px] text-center text-gray-300">
                                {plot.title}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default LatentSpaceVisualizer;
