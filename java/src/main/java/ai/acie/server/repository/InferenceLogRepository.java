package ai.acie.server.repository;

import ai.acie.server.model.InferenceLog;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface InferenceLogRepository extends JpaRepository<InferenceLog, Long> {
    List<InferenceLog> findByStatus(String status);

    List<InferenceLog> findByTimestampAfter(LocalDateTime timestamp);
}
