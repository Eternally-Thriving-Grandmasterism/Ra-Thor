# Powrush Network Prediction for Movement System v14.5

**Client-Side Prediction + Server Reconciliation for Distance-Modulated Jumping**  
**Production Implementation Guide**  
**Aligned with POWRUSH_MOVEMENT_SYSTEM_DESIGN_v14.5**  
**AG-SML v1.0 | TOLC 8 Aligned**

---

## 1. Overview

This document provides a production-grade implementation approach for **client-side prediction** and **server reconciliation** of the Powrush movement system (Conquer Online-style distance-modulated jumping).

**Goals**
- Zero perceived latency for movement on the client.
- Authoritative server state.
- Smooth correction when desync occurs.
- Maintains the expressive, momentum-based kungfu jumping feel.

---

## 2. High-Level Architecture

**Standard Authoritative Server + Client Prediction Model**

1. **Client** predicts movement immediately and renders it.
2. **Client** sends movement intent (target position + timestamp) to server.
3. **Server** simulates the jump using the same deterministic formula.
4. **Server** sends authoritative result back to client.
5. **Client** reconciles if prediction was incorrect.

---

## 3. Client-Side Prediction Implementation

### 3.1 Prediction Flow

```csharp
public class MovementPredictor
{
    private Queue<MovementInput> pendingInputs = new Queue<MovementInput>();
    private Vector3 lastServerPosition;
    private float lastServerTimestamp;

    public void OnPlayerClickMove(Vector3 targetPosition)
    {
        var input = new MovementInput
        {
            Timestamp = Time.time,
            TargetPosition = targetPosition,
            StartingPosition = transform.position
        };

        // 1. Immediately predict and apply on client
        PredictAndApplyJump(input);

        // 2. Send to server
        SendMovementInputToServer(input);

        pendingInputs.Enqueue(input);
    }

    private void PredictAndApplyJump(MovementInput input)
    {
        // Use exact same formula as server
        float distance = Vector3.Distance(input.StartingPosition, input.TargetPosition);
        var jumpParams = CalculateJumpParameters(distance); // Same function as server

        // Apply predicted trajectory
        StartPredictedJump(jumpParams);
    }
}
```

### 3.2 Key Requirements for Determinism

The jump calculation **must be 100% deterministic** between client and server:
- Same `CalculateJumpParameters(distance)` function
- Same gravity multiplier during jumps
- Same animation timing logic
- Fixed-point or consistent floating-point math

---

## 4. Server Simulation

```csharp
public class MovementAuthority
{
    public void OnReceiveMovementInput(Player player, MovementInput input)
    {
        // Validate input (anti-cheat basics)
        if (!ValidateMovementInput(player, input)) return;

        // Simulate jump using same formula
        var jumpParams = CalculateJumpParameters(
            Vector3.Distance(input.StartingPosition, input.TargetPosition));

        // Apply authoritative movement
        player.ApplyAuthoritativeJump(jumpParams);

        // Send result to client
        SendMovementResult(player, jumpParams, input.Timestamp);
    }
}
```

---

## 5. Reconciliation (Client Correction)

When the server sends a correction:

```csharp
public void OnReceiveServerCorrection(MovementResult result)
{
    // Remove acknowledged inputs
    while (pendingInputs.Count > 0 && pendingInputs.Peek().Timestamp <= result.Timestamp)
    {
        pendingInputs.Dequeue();
    }

    // Check if we need to correct
    if (Vector3.Distance(transform.position, result.FinalPosition) > reconciliationThreshold)
    {
        // Option A: Hard snap (simple but can feel jarring)
        // transform.position = result.FinalPosition;

        // Option B: Smooth reconciliation (recommended)
        StartSmoothReconciliation(result.FinalPosition, result.FinalVelocity);
    }

    // Replay remaining predicted inputs on top of corrected state
    ReplayPendingInputs();
}
```

**Smooth Reconciliation** is strongly recommended for Powrush to preserve the fluid kungfu movement feel.

---

## 6. Edge Cases & Considerations

- **Long Jumps**: Higher chance of visible correction due to longer prediction window. Use stronger smoothing.
- **Skill Usage Mid-Jump**: Must predict ability state alongside movement.
- **Terrain Collision**: Server must have final authority on landing position if terrain changes or obstacles appear.
- **Latency Spikes**: Implement input buffering and extrapolation for very high latency.

---

## 7. Recommended Implementation Order

1. Make jump calculation fully deterministic (extract to shared utility).
2. Implement basic client prediction (apply immediately).
3. Implement server simulation + result sending.
4. Implement reconciliation with smoothing.
5. Add input buffering and replay system.
6. Stress test with simulated latency and packet loss.

---

*This network prediction layer is critical for delivering responsive, satisfying kungfu-style movement while maintaining authoritative server control.*
