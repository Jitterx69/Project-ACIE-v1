package ai.acie.server;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;

/**
 * ACIE Distributed Inference Server
 * 
 * Provides:
 * - REST API for counterfactual inference
 * - gRPC service for high-performance calls
 * - Batch processing capabilities
 * - Model version management
 * - Monitoring and metrics
 */
@SpringBootApplication
@EnableAsync
@EnableJpaRepositories
public class ACIEInferenceServer {

    public static void main(String[] args) {
        SpringApplication.run(ACIEInferenceServer.class, args);
    }

    @Bean(name = "inferenceExecutor")
    public Executor inferenceExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(4);
        executor.setMaxPoolSize(16);
        executor.setQueueCapacity(1000);
        executor.setThreadNamePrefix("inference-");
        executor.initialize();
        return executor;
    }
}
