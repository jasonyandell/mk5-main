import type { EventEnvelope } from '../events/types';
import type { EventStore } from '../events/store';

export class TriggerManager {
  constructor(private store: EventStore) {}

  handleEvent(event: EventEnvelope): void {
    switch (event.payload.type) {
      case 'AI_SCHEDULED':
        this.emitAIEvents(event.payload.player);
        break;
    }
  }

  private emitAIEvents(player: number): void {
    this.store.append({ 
      type: 'AI_THINKING', 
      player 
    });
  }
}