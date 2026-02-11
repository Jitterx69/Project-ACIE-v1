; Assembly optimized matrix operations for ACIE
; x86-64 NASM syntax
; macOS Mach-O format compatible
; Provides ultra-fast matrix kernels for critical paths

section .text

; Export symbols with underscore prefix for macOS
global _fast_matrix_multiply_asm
global _fast_relu_asm
global _fast_sigmoid_asm

; Fast matrix multiplication: C = A * B (simplified version)
; Parameters (System V AMD64 ABI / macOS):
;   RDI = A pointer (float*)
;   RSI = B pointer (float*)
;   RDX = C pointer (float*)
;   RCX = M (rows of A)
;   R8  = N (cols of A, rows of B)
;   R9  = K (cols of B)
_fast_matrix_multiply_asm:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15
    
    ; Save parameters
    mov r12, rdi    ; A
    mov r13, rsi    ; B
    mov r14, rdx    ; C
    mov r15, rcx    ; M
    
    ; Simple non-vectorized implementation for compatibility
    ; Outer loop: rows of A
    xor r10, r10    ; i = 0
.outer_loop:
    cmp r10, r15
    jge .done
    
    ; Middle loop: cols of B
    xor r11, r11    ; j = 0
.middle_loop:
    cmp r11, r9
    jge .outer_next
    
    ; Initialize sum
    xorps xmm0, xmm0
    
    ; Inner loop: dot product
    xor rbx, rbx    ; k = 0
.inner_loop:
    cmp rbx, r8
    jge .store_result
    
    ; Load A[i][k]
    mov rax, r10
    imul rax, r8
    add rax, rbx
    shl rax, 2
    movss xmm1, [r12 + rax]
    
    ; Load B[k][j]
    mov rax, rbx
    imul rax, r9
    add rax, r11
    shl rax, 2
    movss xmm2, [r13 + rax]
    
    ; Multiply and accumulate
    mulss xmm1, xmm2
    addss xmm0, xmm1
    
    inc rbx
    jmp .inner_loop
    
.store_result:
    ; Store result in C[i][j]
    mov rax, r10
    imul rax, r9
    add rax, r11
    shl rax, 2
    movss [r14 + rax], xmm0
    
    inc r11
    jmp .middle_loop
    
.outer_next:
    inc r10
    jmp .outer_loop
    
.done:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; Fast ReLU: out[i] = max(0, in[i])
; Parameters:
;   RDI = input pointer
;   RSI = output pointer
;   RDX = length
_fast_relu_asm:
    push rbp
    mov rbp, rsp
    
    xorps xmm1, xmm1  ; Zero register
    xor rcx, rcx
    
.loop:
    cmp rcx, rdx
    jge .done
    
    ; Load float
    mov rax, rcx
    shl rax, 2
    movss xmm0, [rdi + rax]
    
    ; Max with zero
    maxss xmm0, xmm1
    
    ; Store
    movss [rsi + rax], xmm0
    
    inc rcx
    jmp .loop
    
.done:
    pop rbp
    ret

; Fast sigmoid approximation
; Approximation: σ(x) ≈ 0.5 * (x / (1 + |x|)) + 0.5
; Parameters:
;   RDI = input pointer
;   RSI = output pointer
;   RDX = length
_fast_sigmoid_asm:
    push rbp
    mov rbp, rsp
    push rbx
    
    ; Load constants
    mov eax, 0x3f000000  ; 0.5
    movd xmm7, eax
    mov eax, 0x3f800000  ; 1.0
    movd xmm6, eax
    mov eax, 0x7FFFFFFF  ; abs mask
    movd xmm5, eax
    
    xor rcx, rcx
    
.loop:
    cmp rcx, rdx
    jge .done
    
    ; Load x
    mov rax, rcx
    shl rax, 2
    movss xmm0, [rdi + rax]
    
    ; Compute |x|
    movaps xmm1, xmm0
    andps xmm1, xmm5
    
    ; 1 + |x|
    movaps xmm2, xmm6
    addss xmm2, xmm1
    
    ; x / (1 + |x|)
    divss xmm0, xmm2
    
    ; 0.5 * result
    mulss xmm0, xmm7
    
    ; + 0.5
    addss xmm0, xmm7
    
    ; Store
    movss [rsi + rax], xmm0
    
    inc rcx
    jmp .loop
    
.done:
    pop rbx
    pop rbp
    ret

