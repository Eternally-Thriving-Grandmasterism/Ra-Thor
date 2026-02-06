// src/core/event-sourcing-integration.ts – Event Sourcing Integration Layer v1.1
// Full CQRS + Event Sourcing bridge: event-sourced write model + projected read models
// Valence-aware event validation, snapshotting, mercy-protected replay & commit
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { EventEmitter } from 'events';

// ──────────────────────────────────────────────────────────────
// Domain Event definition (immutable, append-only)
// ──────────────────────────────────────────────────────────────

export interface DomainEvent<T = any> {
  type: string;                           // e.g. 'ValenceUpdated', 'GestureRecognized'
  aggregateId: string;                    // e.g. 'global-valuation', 'user-uuid'
  timestamp: number;
  version: number;                        // per-aggregate event version
  payload: T;
  valence: number;                        // valence at event creation time
  correlationId?: string;                 // for tracing sagas/transactions
  causationId?: string;                   // parent event id
}

// ──────────────────────────────────────────────────────────────
// Aggregate root interface (write model)
// ──────────────────────────────────────────────────────────────

export interface AggregateRoot<TState> {
  aggregateId: string;
  version: number;
  state: TState;
  uncommittedEvents: DomainEvent[];
  apply(event: DomainEvent): void;
  loadFromHistory(events: DomainEvent[]): void;
  raiseEvent(type: string, payload: any): void;
}

// ──────────────────────────────────────────────────────────────
// In-memory event store (replace with Redis/Kafka/DB in production)
// ──────────────────────────────────────────────────────────────

class InMemoryEventStore {
  private events: Map<string, DomainEvent[]> = new Map();
  private snapshots: Map<string, { state: any; version: number; timestamp: number }> = new Map();

  async append(event: DomainEvent): Promise<void> {
    if (!this.events.has(event.aggregateId)) {
      this.events.set(event.aggregateId, []);
    }
    const aggregateEvents = this.events.get(event.aggregateId)!;
    if (event.version !== aggregateEvents.length + 1) {
      throw new Error(`Version conflict for ${event.aggregateId}: expected ${aggregateEvents.length + 1}, got ${event.version}`);
    }
    aggregateEvents.push(event);
    eventBus.emit('EVENT_APPENDED', event);
  }

  async getEvents(aggregateId: string, fromVersion = 0): Promise<DomainEvent[]> {
    const events = this.events.get(aggregateId) || [];
    return events.slice(fromVersion);
  }

  async getSnapshot(aggregateId: string): Promise<{ state: any; version: number; timestamp: number } | null> {
    return this.snapshots.get(aggregateId) || null;
  }

  async saveSnapshot(aggregateId: string, state: any, version: number): Promise<void> {
    this.snapshots.set(aggregateId, { state, version, timestamp: Date.now() });
  }
}

const eventStore = new InMemoryEventStore();
const eventBus = new EventEmitter();

// ──────────────────────────────────────────────────────────────
// Read model projectors (in-memory for now – replace with Redis/DB)
// ──────────────────────────────────────────────────────────────

interface ReadModel<T = any> {
  [aggregateId: string]: T;
}

const readModels: { [modelName: string]: ReadModel } = {
  valence: {},      // { aggregateId: { currentValence, history, lastUpdated } }
  gesture: {},      // { aggregateId: { lastGesture, confidence, timestamp } }
  // Add more read models as needed
};

function projectEvent(event: DomainEvent) {
  const { type, aggregateId, payload, valence, timestamp } = event;

  switch (type) {
    case 'ValenceUpdated':
      readModels.valence[aggregateId] = {
        currentValence: payload.newValence,
        history: [...(readModels.valence[aggregateId]?.history || []), payload.newValence],
        lastUpdated: timestamp,
        lastValence: valence
      };
      break;

    case 'GestureRecognized':
      readModels.gesture[aggregateId] = {
        lastGesture: payload.gesture,
        confidence: payload.confidence,
        timestamp,
        valence
      };
      break;

    default:
      console.debug(`[CQRS] No projection handler for event type: ${type}`);
  }
}

// Subscribe to event appends
eventBus.on('EVENT_APPENDED', projectEvent);

// ──────────────────────────────────────────────────────────────
// Aggregate base class (write model)
// ──────────────────────────────────────────────────────────────

abstract class AggregateRootBase<TState> implements AggregateRoot<TState> {
  aggregateId: string;
  version = 0;
  state: TState;
  uncommittedEvents: DomainEvent[] = [];

  constructor(aggregateId: string, initialState: TState) {
    this.aggregateId = aggregateId;
    this.state = initialState;
  }

  protected raiseEvent(type: string, payload: any) {
    const valence = currentValence.get();

    const event: DomainEvent = {
      type,
      aggregateId: this.aggregateId,
      timestamp: Date.now(),
      version: this.version + 1,
      payload,
      valence
    };

    this.apply(event);
    this.uncommittedEvents.push(event);
  }

  loadFromHistory(events: DomainEvent[]) {
    for (const event of events) {
      this.apply(event);
      this.version = Math.max(this.version, event.version);
    }
  }

  abstract apply(event: DomainEvent): void;

  async commit(): Promise<void> {
    if (!await mercyGate(`Commit aggregate ${this.aggregateId}`)) {
      throw new Error('Mercy gate blocked commit');
    }

    for (const event of this.uncommittedEvents) {
      await eventStore.append(event);
    }

    // Optional snapshot on high-version intervals
    if (this.version % 50 === 0) {
      await eventStore.saveSnapshot(this.aggregateId, this.state, this.version);
    }

    this.uncommittedEvents = [];
  }
}

// ──────────────────────────────────────────────────────────────
// Example Aggregate: Global Valence Tracker
// ──────────────────────────────────────────────────────────────

interface ValenceState {
  currentValence: number;
  history: { timestamp: number; valence: number }[];
  lastUpdated: number;
}

class GlobalValenceAggregate extends AggregateRootBase<ValenceState> {
  constructor(aggregateId: string = 'global-valuation') {
    super(aggregateId, {
      currentValence: 0.5,
      history: [],
      lastUpdated: Date.now()
    });
  }

  apply(event: DomainEvent) {
    switch (event.type) {
      case 'ValenceUpdated':
        this.state.currentValence = event.payload.newValence;
        this.state.history.push({ timestamp: event.timestamp, valence: event.payload.newValence });
        this.state.lastUpdated = event.timestamp;
        break;
      default:
        console.warn(`Unknown event type: ${event.type}`);
    }
  }

  updateValence(newValence: number) {
    if (newValence < 0 || newValence > 1) {
      throw new Error('Valence must be between 0 and 1');
    }
    this.raiseEvent('ValenceUpdated', { newValence });
  }
}

// ──────────────────────────────────────────────────────────────
// CQRS Command & Query Dispatchers
// ──────────────────────────────────────────────────────────────

export async function dispatchCommand<TCommand extends { type: string; aggregateId: string }>(
  command: TCommand
): Promise<void> {
  const actionName = `Dispatch command: ${command.type}`;
  if (!await mercyGate(actionName)) {
    throw new Error(`Mercy gate blocked command: ${command.type}`);
  }

  const valence = currentValence.get();

  // Load aggregate (example – generalize for different aggregates)
  const aggregate = await loadAggregate(command.aggregateId, GlobalValenceAggregate);

  // Handle command (expand with switch or handler map)
  if (command.type === 'UpdateValence') {
    aggregate.updateValence((command as any).payload.newValence);
  } else {
    throw new Error(`Unknown command type: ${command.type}`);
  }

  await aggregate.commit();

  mercyHaptic.playPattern('cosmicHarmony', valence);
}

// ──────────────────────────────────────────────────────────────
// CQRS Query Dispatcher
// ──────────────────────────────────────────────────────────────

export async function dispatchQuery<TResult = any>(
  queryType: string,
  payload: any = {}
): Promise<TResult> {
  const actionName = `Dispatch query: ${queryType}`;
  if (!await mercyGate(actionName)) {
    throw new Error(`Mercy gate blocked query: ${queryType}`);
  }

  switch (queryType) {
    case 'GetCurrentValence':
      const aggregateId = payload.aggregateId || 'global-valuation';
      const state = readModels.valence[aggregateId];
      if (!state) throw new Error(`Valence state not found for ${aggregateId}`);
      return state.currentValence as TResult;

    default:
      throw new Error(`Unknown query type: ${queryType}`);
  }
}

// ──────────────────────────────────────────────────────────────
// Helper: Load aggregate with snapshot optimization
// ──────────────────────────────────────────────────────────────

async function loadAggregate<T extends AggregateRoot<any>>(
  aggregateId: string,
  aggregateClass: new (id: string) => T
): Promise<T> {
  const snapshot = await eventStore.getSnapshot(aggregateId);
  let aggregate: T;

  if (snapshot) {
    aggregate = new aggregateClass(aggregateId);
    aggregate.state = snapshot.state;
    aggregate.version = snapshot.version;
  } else {
    aggregate = new aggregateClass(aggregateId);
  }

  const events = await eventStore.getEvents(aggregateId, aggregate.version);
  aggregate.loadFromHistory(events);

  return aggregate;
}

// ──────────────────────────────────────────────────────────────
// Example CQRS usage
// ──────────────────────────────────────────────────────────────

export async function updateAndQueryValence(newValence: number) {
  const command = {
    type: 'UpdateValence',
    aggregateId: 'global-valuation',
    payload: { newValence }
  };

  await dispatchCommand(command);
  const current = await dispatchQuery<number>('GetCurrentValence');
  console.log(`Global valence updated & queried: ${current}`);
}
