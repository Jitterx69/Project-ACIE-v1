package ai.acie.server.config;

import ai.acie.server.security.JwtTokenProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.filter.OncePerRequestFilter;

import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity
public class SecurityConfig {

    private final JwtTokenProvider tokenProvider;

    public SecurityConfig(JwtTokenProvider tokenProvider) {
        this.tokenProvider = tokenProvider;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                .csrf(csrf -> csrf.disable())
                .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers("/api/auth/**", "/actuator/**").permitAll()
                        .anyRequest().authenticated())
                .addFilterBefore(new JwtAuthenticationFilter(tokenProvider),
                        UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    // Inner class for simplicity, usually separate file
    public static class JwtAuthenticationFilter extends OncePerRequestFilter {
        private final JwtTokenProvider tokenProvider;

        public JwtAuthenticationFilter(JwtTokenProvider tokenProvider) {
            this.tokenProvider = tokenProvider;
        }

        @Override
        protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                FilterChain filterChain)
                throws ServletException, IOException {
            try {
                String jwt = getJwtFromRequest(request);
                if (jwt != null && tokenProvider.validateToken(jwt)) {
                    String username = tokenProvider.getUsernameFromJWT(jwt);

                    // Create simple authentication (no roles for now)
                    UsernamePasswordAuthenticationToken authentication = new UsernamePasswordAuthenticationToken(
                            username, null, java.util.Collections.emptyList());

                    authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                    SecurityContextHolder.getContext().setAuthentication(authentication);
                }
            } catch (Exception ex) {
                // log error
            }

            filterChain.doFilter(request, response);
        }

        private String getJwtFromRequest(HttpServletRequest request) {
            String bearerToken = request.getHeader("Authorization");
            if (bearerToken != null && bearerToken.startsWith("Bearer ")) {
                return bearerToken.substring(7);
            }
            return null;
        }
    }
}
