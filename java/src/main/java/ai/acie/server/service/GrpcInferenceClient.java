package ai.acie.server.service;

import acie.InferenceServiceGrpc;
import acie.Acie;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.PreDestroy;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
public class GrpcInferenceClient {

    private final ManagedChannel channel;
    private final InferenceServiceGrpc.InferenceServiceBlockingStub blockingStub;

    public GrpcInferenceClient(@Value("${acie.inference.host:localhost}") String host,
            @Value("${acie.inference.port:50051}") int port) {
        this.channel = ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext() // Disable TLS for internal communication
                .build();
        this.blockingStub = InferenceServiceGrpc.newBlockingStub(channel);
    }

    public Acie.InferenceResponse infer(String modelVersion, List<Float> observation, Map<String, Float> intervention,
            String requestId) {
        Acie.InferenceRequest request = Acie.InferenceRequest.newBuilder()
                .setModelVersion(modelVersion)
                .addAllObservation(observation)
                .putAllIntervention(intervention)
                .setRequestId(requestId)
                .build();

        return blockingStub.counterfactualInference(request);
    }

    @PreDestroy
    public void shutdown() throws InterruptedException {
        channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
    }
}
