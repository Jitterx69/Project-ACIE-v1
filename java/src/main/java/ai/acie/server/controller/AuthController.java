package ai.acie.server.controller;

import ai.acie.server.security.JwtTokenProvider;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final JwtTokenProvider tokenProvider;

    public AuthController(JwtTokenProvider tokenProvider) {
        this.tokenProvider = tokenProvider;
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@RequestBody Map<String, String> credentials) {
        // Mock authentication - accept any user for demo
        String username = credentials.get("username");
        if (username == null || username.isEmpty()) {
            return ResponseEntity.badRequest().body("Username required");
        }

        String token = tokenProvider.generateToken(username);
        return ResponseEntity.ok(Collections.singletonMap("token", token));
    }
}
