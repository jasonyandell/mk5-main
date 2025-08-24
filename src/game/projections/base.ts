import type { EventEnvelope, GameEvent } from '../events/types';
import { writable, type Writable, type Readable, derived } from 'svelte/store';

export abstract class Projection<T> {
  protected store: Writable<T>;
  public readonly value: Readable<T>;
  protected currentValue: T;

  constructor(initialValue: T) {
    this.currentValue = initialValue;
    this.store = writable(initialValue);
    this.value = this.store;
  }

  handleEvent(event: EventEnvelope): void {
    if (event.payload.type === 'RESET') {
      this.currentValue = this.getInitialValue();
      this.store.set(this.currentValue);
      return;
    }

    const newValue = this.project(this.currentValue, event.payload);
    if (newValue !== this.currentValue) {
      this.currentValue = newValue;
      this.store.set(newValue);
    }
  }

  protected abstract getInitialValue(): T;
  protected abstract project(current: T, event: GameEvent): T;

  getCurrentValue(): T {
    return this.currentValue;
  }
}