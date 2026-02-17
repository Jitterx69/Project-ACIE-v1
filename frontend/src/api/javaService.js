export const fetchAuditLogs = async () => {
    try {
        // In a real app, this would be: await fetch('/api/audit/logs');
        // Returning mock data for now since the backend integration isn't live without compilation
        return [
            { id: 1, timestamp: new Date().toISOString(), status: 'SUCCESS', latency: 45, inputHash: 'a1b2c3d4' },
            { id: 2, timestamp: new Date(Date.now() - 5000).toISOString(), status: 'SUCCESS', latency: 42, inputHash: 'e5f6g7h8' },
            { id: 3, timestamp: new Date(Date.now() - 15000).toISOString(), status: 'FAILURE', latency: 120, inputHash: 'i9j0k1l2' },
            { id: 4, timestamp: new Date(Date.now() - 30000).toISOString(), status: 'SUCCESS', latency: 50, inputHash: 'm3n4o5p6' },
        ];
    } catch (error) {
        console.error("Failed to fetch audit logs", error);
        return [];
    }
};
