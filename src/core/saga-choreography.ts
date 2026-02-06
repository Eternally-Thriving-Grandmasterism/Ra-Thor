// src/core/saga-choreography.ts – Saga Choreography Pattern v1.0
// Decentralized event-driven saga implementation – no central orchestrator
// Each participant handles its own local transaction + compensation
// Valence-aware event routing, mercy-protected rollback, lattice-integrated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { EventEmitter } from 'events';

// ──────────────────────────────────────────────────────────────
// Event types & payload structure
// ──────────────────────────────────────────────────────────────

type SagaEventType =
  | 'RESERVE_RESOURCES_STARTED'
  | 'RESERVE_RESOURCES_COMPLETED'
  | 'RESERVE_RESOURCES_FAILED'
  | 'PROCESS_PAYMENT_STARTED'
  | 'PROCESS_PAYMENT_COMPLETED'
  | 'PROCESS_PAYMENT_FAILED'
  | 'SHIP_ORDER_STARTED'
  | 'SHIP_ORDER_COMPLETED'
  | 'SHIP_ORDER_FAILED'
  | 'SAGA_ABORTED';

interface SagaEvent<T = any> {
  type: SagaEventType;
  correlationId: string;
  timestamp: number;
  payload: T;
  valence: number;
  source: string;                     // service/module name
}

interface SagaParticipant {
  name: string;
  handleEvent: (event: SagaEvent) => Promise<void>;
  compensate: (event: SagaEvent) => Promise<void>;
}

// ──────────────────────────────────────────────────────────────
// Central event bus (lightweight – can be replaced with Redis/Kafka)
const eventBus = new EventEmitter();

// ──────────────────────────────────────────────────────────────
// Participant registry
const participants = new Map<string, SagaParticipant>();

// ──────────────────────────────────────────────────────────────
// Core choreography engine
export class SagaChoreography {
  private correlationId: string;
  private completedSteps = new Set<string>();
  private failedSteps = new Set<string>();

  constructor(correlationId: string = crypto.randomUUID()) {
    this.correlationId = correlationId;
  }

  /**
   * Register a participant service/module
   */
  static registerParticipant(participant: SagaParticipant) {
    participants.set(participant.name, participant);
    console.log(`[SagaChoreo] Registered participant: ${participant.name}`);
  }

  /**
   * Start a new saga instance
   */
  async start(initialEvent: Omit<SagaEvent, 'correlationId' | 'timestamp' | 'valence'>) {
    const actionName = 'Start saga choreography';
    if (!await mercyGate(actionName)) {
      throw new Error('Mercy gate blocked saga start');
    }

    const valence = currentValence.get();
    const event: SagaEvent = {
      ...initialEvent,
      correlationId: this.correlationId,
      timestamp: Date.now(),
      valence,
      source: 'saga-initiator'
    };

    console.log(`[Saga:${this.correlationId}] Starting – type: ${event.type}, valence: ${valence.toFixed(3)}`);

    await this.emitEvent(event);
  }

  /**
   * Emit event to all relevant participants
   */
  private async emitEvent(event: SagaEvent) {
    eventBus.emit(event.type, event);

    // Also emit to correlation-specific listeners if any
    eventBus.emit(`saga:\( {this.correlationId}: \){event.type}`, event);

    // Log for traceability
    console.log(`[Saga:${this.correlationId}] Emitted event: ${event.type} from ${event.source}`);
  }

  /**
   * Listen for events and route to local handler
   */
  listen() {
    // Listen to all saga events
    eventBus.on('RESERVE_RESOURCES_STARTED', this.handleEvent.bind(this));
    eventBus.on('PROCESS_PAYMENT_STARTED', this.handleEvent.bind(this));
    eventBus.on('SHIP_ORDER_STARTED', this.handleEvent.bind(this));

    // Listen to failure events for compensation
    eventBus.on('RESERVE_RESOURCES_FAILED', this.handleCompensation.bind(this));
    eventBus.on('PROCESS_PAYMENT_FAILED', this.handleCompensation.bind(this));
    eventBus.on('SHIP_ORDER_FAILED', this.handleCompensation.bind(this));
  }

  private async handleEvent(event: SagaEvent) {
    const participant = participants.get(event.source);
    if (!participant) return;

    try {
      await participant.handleEvent(event);
      this.completedSteps.add(event.type);

      // Trigger next step if previous completed
      await this.triggerNextStep(event);
    } catch (err) {
      this.failedSteps.add(event.type);
      await this.emitEvent({
        type: `${event.type}_FAILED` as SagaEventType,
        correlationId: event.correlationId,
        timestamp: Date.now(),
        payload: err,
        valence: currentValence.get(),
        source: event.source
      });
    }
  }

  private async handleCompensation(event: SagaEvent) {
    const participant = participants.get(event.source);
    if (!participant || !participant.compensate) return;

    try {
      await participant.compensate(event);
      console.log(`[Saga:${event.correlationId}] Compensated: ${event.source}`);
    } catch (compErr) {
      console.error(`[Saga:${event.correlationId}] Compensation failed for ${event.source}:`, compErr);
      // Continue best-effort compensation
    }
  }

  private async triggerNextStep(completedEvent: SagaEvent) {
    const valence = currentValence.get();

    // Simple linear workflow – customize for complex sagas
    switch (completedEvent.type) {
      case 'RESERVE_RESOURCES_COMPLETED':
        await this.emitEvent({
          type: 'PROCESS_PAYMENT_STARTED',
          correlationId: this.correlationId,
          timestamp: Date.now(),
          payload: completedEvent.payload,
          valence,
          source: 'payment-service'
        });
        break;

      case 'PROCESS_PAYMENT_COMPLETED':
        await this.emitEvent({
          type: 'SHIP_ORDER_STARTED',
          correlationId: this.correlationId,
          timestamp: Date.now(),
          payload: completedEvent.payload,
          valence,
          source: 'shipping-service'
        });
        break;

      case 'SHIP_ORDER_COMPLETED':
        console.log(`[Saga:${this.correlationId}] Fully completed`);
        mercyHaptic.playPattern('cosmicHarmony', valence);
        break;
    }
  }
}

/**
 * Global singleton orchestrator instance
 */
export const sagaChoreography = new SagaChoreography();

// Register built-in participants (expand with real services)
sagaChoreography['participants'] = new Map();

// Example participant registration (in real code, services register themselves)
SagaChoreography.registerParticipant({
  name: 'reservation-service',
  handleEvent: async (event) => {
    if (event.type === 'RESERVE_RESOURCES_STARTED') {
      // Simulate work
      await new Promise(r => setTimeout(r, 800));
      console.log('[Reservation] Resources reserved');
    }
  },
  compensate: async (event) => {
    console.log('[Reservation] Compensating reservation');
  }
});

// More participants...

// Start listening
sagaChoreography.listen();
