export const fetchAnalysisPlots = async () => {
    // Mock data for R plots
    return [
        { id: 1, title: "Latent Space (PCA)", url: "https://via.placeholder.com/400x300?text=PCA+Plot" },
        { id: 2, title: "Latent Space (t-SNE)", url: "https://via.placeholder.com/400x300?text=t-SNE+Plot" },
        { id: 3, title: "Spatial Correlation", url: "https://via.placeholder.com/400x300?text=2PCF+Plot" },
    ];
};
