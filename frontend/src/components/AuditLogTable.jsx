import React, { useState, useEffect } from 'react';
import { fetchAuditLogs } from '../api/javaService';
import { CheckCircle, XCircle, Clock, Hash } from 'lucide-react';

const AuditLogTable = () => {
    const [logs, setLogs] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadLogs = async () => {
            const data = await fetchAuditLogs();
            setLogs(data);
            setLoading(false);
        };

        loadLogs();
        const interval = setInterval(loadLogs, 5000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="bg-surface border border-gray-800 rounded-xl p-6 shadow-lg overflow-hidden">
            <h3 className="text-lg font-semibold mb-4 text-white flex items-center gap-2">
                <Clock className="w-5 h-5 text-blue-400" />
                Live Audit Logs
            </h3>
            {loading ? (
                <div className="flex justify-center p-8 text-gray-500">
                    <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                </div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse">
                        <thead>
                            <tr className="text-gray-500 text-xs border-b border-gray-800">
                                <th className="p-3">Status</th>
                                <th className="p-3">Timestamp</th>
                                <th className="p-3">Latency</th>
                                <th className="p-3">Input Hash</th>
                            </tr>
                        </thead>
                        <tbody className="text-sm">
                            {logs.map((log) => (
                                <tr key={log.id} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                                    <td className="p-3 flex items-center gap-2">
                                        {log.status === 'SUCCESS' ? (
                                            <CheckCircle className="w-4 h-4 text-green-500" />
                                        ) : (
                                            <XCircle className="w-4 h-4 text-red-500" />
                                        )}
                                        <span className={log.status === 'SUCCESS' ? 'text-green-400' : 'text-red-400'}>
                                            {log.status}
                                        </span>
                                    </td>
                                    <td className="p-3 text-gray-400">
                                        {new Date(log.timestamp).toLocaleTimeString()}
                                    </td>
                                    <td className="p-3 text-white font-mono">
                                        {log.latency}ms
                                    </td>
                                    <td className="p-3 text-gray-500 font-mono text-xs flex items-center gap-1">
                                        <Hash className="w-3 h-3" />
                                        {log.inputHash}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default AuditLogTable;
