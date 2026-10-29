// src/core/saga-orchestrator.ts – Saga Pattern Orchestrator v1.0
// Centralized saga orchestration for distributed transactions
// Step-by-step execution with compensation on failure, valence-aware rollback gating
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

type SagaStep<TInput = any, TOutput = any> = {
  name: string;
  execute: (input: TInput) => Promise<TOutput>;
  compensate?: (input: TInput, output: TOutput | Error) => Promise<void>;
  timeoutMs?: number;
  requiredValence?: number;         // min valence to attempt this step
};

interface SagaContext {
  correlationId: string;
  currentStepIndex: number;
  results: any[];
  errors: Error[];
}

export class SagaOrchestrator {
  private steps: SagaStep[] = [];
  private context: SagaContext | null = null;

  addStep<TInput, TOutput>(step: SagaStep<TInput, TOutput>): this {
    this.steps.push(step);
    return this;
  }

  /**
   * Execute the saga with compensation on failure
   * @param initialInput Starting input for the first step
   * @returns Final output or throws if saga fails and compensation cannot recover
   */
  async execute<TInput, TFinalOutput>(initialInput: TInput): Promise<TFinalOutput> {
    const actionName = 'Execute saga orchestration';
    if (!await mercyGate(actionName)) {
      throw new Error('Mercy gate blocked saga execution');
    }

    const valence = currentValence.get();
    const correlationId = crypto.randomUUID();

    this.context = {
      correlationId,
      currentStepIndex: 0,
      results: [],
      errors: []
    };

    console.log(`[Saga:${correlationId}] Starting saga – valence ${valence.toFixed(3)}`);

    try {
      let currentInput: any = initialInput;

      for (let i = 0; i < this.steps.length; i++) {
        const step = this.steps[i];
        this.context.currentStepIndex = i;

        // Valence gate for this step
        if (step.requiredValence && valence < step.requiredValence) {
          throw new Error(`Insufficient valence (\( {valence.toFixed(3)}) for step " \){step.name}" (requires ≥${step.requiredValence})`);
        }

        console.log(`[Saga:${correlationId}] Executing step \( {i+1}/ \){this.steps.length}: ${step.name}`);

        try {
          const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`Step "${step.name}" timed out after ${step.timeoutMs || 30000}ms`)), step.timeoutMs || 30000)
          );

          const output = await Promise.race([
            step.execute(currentInput),
            timeoutPromise
          ]) as TOutput;

          this.context.results.push(output);
          currentInput = output; // chain output to next step input

        } catch (error) {
          this.context.errors.push(error as Error);
          console.error(`[Saga:\( {correlationId}] Step " \){step.name}" failed:`, error);

          // Compensation phase – rollback previous successful steps
          await this.compensate(i);
          throw error; // re-throw to propagate failure
        }
      }

      mercyHaptic.playPattern('cosmicHarmony', valence);
      console.log(`[Saga:${correlationId}] Completed successfully`);
      return this.context.results[this.context.results.length - 1] as TFinalOutput;

    } catch (finalError) {
      mercyHaptic.playPattern('warningPulse', currentValence.get() * 0.7);
      console.error(`[Saga:${correlationId}] Failed after ${this.context.currentStepIndex + 1} steps`);
      throw finalError;
    } finally {
      this.context = null; // cleanup
    }
  }

  /**
   * Compensate (rollback) steps from last successful to first
   */
  private async compensate(failedIndex: number): Promise<void> {
    if (!this.context) return;

    const valence = currentValence.get();
    console.log(`[Saga:${this.context.correlationId}] Starting compensation from step ${failedIndex}`);

    for (let i = failedIndex - 1; i >= 0; i--) {
      const step = this.steps[i];
      if (!step.compensate) continue;

      try {
        const previousOutput = this.context.results[i];
        await step.compensate(this.context.results[i-1] || {}, previousOutput);
        console.log(`[Saga:${this.context.correlationId}] Compensated step ${i+1}: ${step.name}`);
      } catch (compError) {
        console.error(`[Saga:${this.context.correlationId}] Compensation failed for step ${step.name}:`, compError);
        // Continue compensation even if one fails – best effort
      }
    }
  }
}

/**
 * Factory helper – create a saga with fluent API
 */
export function createSaga(): SagaOrchestrator {
  return new SagaOrchestrator();
}

// Example usage: distributed transaction across services
/*
const saga = createSaga()
  .addStep({
    name: 'ReserveResources',
    execute: async (input: { userId: string; amount: number }) => {
      // Call reservation service
      return { reservationId: 'res-123' };
    },
    compensate: async (input, output) => {
      // Cancel reservation
      console.log('Compensating reservation:', output.reservationId);
    },
    requiredValence: 0.9
  })
  .addStep({
    name: 'ProcessPayment',
    execute: async (input: { reservationId: string }) => {
      // Call payment service
      return { paymentId: 'pay-456' };
    },
    compensate: async (input, output) => {
      // Refund payment
      console.log('Refunding payment:', output.paymentId);
    }
  })
  .addStep({
    name: 'ShipOrder',
    execute: async (input: { paymentId: string }) => {
      // Call shipping service
      return { trackingId: 'track-789' };
    }
    // no compensation needed – final step
  });

try {
  const result = await saga.execute({ userId: 'u123', amount: 100 });
  console.log('Saga completed:', result);
} catch (err) {
  console.error('Saga failed and compensated:', err);
}
*/
