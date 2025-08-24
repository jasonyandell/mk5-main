import type { EventEnvelope, GameEvent, ResetEvent } from './types';
import { createHash } from '../core/random';

export class EventStore {
  private events: EventEnvelope[] = [];
  private subscribers: Set<(event: EventEnvelope) => void> = new Set();
  private nextIdx = 0;

  constructor() {}

  static fromURL(): EventStore {
    const store = new EventStore();
    const urlParams = new URLSearchParams(window.location.search);
    
    if (urlParams.has('g')) {
      store.loadFromCompressedString(urlParams.get('g')!);
    } else {
      const chunks: string[] = [];
      let i = 1;
      while (urlParams.has(`g${i}`)) {
        chunks.push(urlParams.get(`g${i}`)!);
        i++;
      }
      if (chunks.length > 0) {
        store.loadFromCompressedString(chunks.join(''));
      }
    }
    
    return store;
  }

  private generateEventId(payload: GameEvent, idx: number): string {
    const content = JSON.stringify(payload) + ':' + idx;
    const hash = createHash(content);
    return hash.toString(36).substring(0, 12);
  }

  append(event: GameEvent, correlationId?: string, causationId?: string): EventEnvelope {
    const envelope: EventEnvelope = {
      id: this.generateEventId(event, this.nextIdx),
      idx: this.nextIdx++,
      timestamp: Date.now(),
      correlationId: correlationId || this.generateCorrelationId(),
      causationId,
      payload: event
    };

    this.events.push(envelope);
    this.notifySubscribers(envelope);
    this.persistToURL();
    
    return envelope;
  }

  appendMultiple(events: GameEvent[], correlationId?: string): EventEnvelope[] {
    const corrId = correlationId || this.generateCorrelationId();
    const envelopes: EventEnvelope[] = [];
    let lastId: string | undefined;

    for (const event of events) {
      const envelope = this.append(event, corrId, lastId);
      envelopes.push(envelope);
      lastId = envelope.id;
    }

    return envelopes;
  }

  getEvents(): EventEnvelope[] {
    return [...this.events];
  }

  getEventAt(index: number): EventEnvelope | undefined {
    return this.events[index];
  }

  getEventCount(): number {
    return this.events.length;
  }

  subscribe(callback: (event: EventEnvelope) => void): () => void {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  private notifySubscribers(event: EventEnvelope): void {
    this.subscribers.forEach(callback => callback(event));
  }

  timeTravel(index: number): void {
    if (index < 0 || index >= this.events.length) return;
    
    this.events = this.events.slice(0, index + 1);
    this.nextIdx = this.events.length;
    
    const resetEvent: ResetEvent = { type: 'RESET' };
    const resetEnvelope: EventEnvelope = {
      id: 'reset',
      idx: -1,
      timestamp: Date.now(),
      correlationId: 'reset',
      payload: resetEvent
    };
    
    this.notifySubscribers(resetEnvelope);
    
    for (const event of this.events) {
      this.notifySubscribers(event);
    }
    
    this.persistToURL();
  }

  clear(): void {
    this.events = [];
    this.nextIdx = 0;
    
    const resetEvent: ResetEvent = { type: 'RESET' };
    const resetEnvelope: EventEnvelope = {
      id: 'reset',
      idx: -1,
      timestamp: Date.now(),
      correlationId: 'reset',
      payload: resetEvent
    };
    
    this.notifySubscribers(resetEnvelope);
    this.persistToURL();
  }

  private generateCorrelationId(): string {
    return Math.random().toString(36).substring(2, 15);
  }

  private persistToURL(): void {
    const compressed = this.compressEvents();
    const urlParams = new URLSearchParams();
    
    const MAX_CHUNK_SIZE = 1900;
    if (compressed.length <= MAX_CHUNK_SIZE) {
      urlParams.set('g', compressed);
    } else {
      const chunks = this.chunkString(compressed, MAX_CHUNK_SIZE);
      chunks.forEach((chunk, i) => {
        urlParams.set(`g${i + 1}`, chunk);
      });
    }
    
    const newURL = `${window.location.pathname}?${urlParams.toString()}`;
    window.history.replaceState({}, '', newURL);
  }

  private compressEvents(): string {
    if (this.events.length === 0) return '';
    
    const minimal = this.events.map(e => ({
      t: e.payload.type,
      p: this.stripDefaults(e.payload)
    }));
    
    const json = JSON.stringify(minimal);
    
    try {
      const encoder = new TextEncoder();
      const data = encoder.encode(json);
      const compressed = this.lzCompress(data);
      return btoa(String.fromCharCode(...compressed)).replace(/[+/=]/g, c => 
        c === '+' ? '-' : c === '/' ? '_' : ''
      );
    } catch {
      return btoa(json).replace(/[+/=]/g, c => 
        c === '+' ? '-' : c === '/' ? '_' : ''
      );
    }
  }

  private loadFromCompressedString(compressed: string): void {
    if (!compressed) return;
    
    try {
      const base64 = compressed.replace(/[-_]/g, c => c === '-' ? '+' : '/');
      const decoded = atob(base64);
      
      let json: string;
      try {
        const bytes = new Uint8Array(decoded.split('').map(c => c.charCodeAt(0)));
        const decompressed = this.lzDecompress(bytes);
        const decoder = new TextDecoder();
        json = decoder.decode(decompressed);
      } catch {
        json = decoded;
      }
      
      const minimal = JSON.parse(json);
      
      this.events = [];
      this.nextIdx = 0;
      
      for (const item of minimal) {
        const event = this.restoreDefaults(item.t, item.p);
        const envelope: EventEnvelope = {
          id: this.generateEventId(event, this.nextIdx),
          idx: this.nextIdx++,
          timestamp: Date.now(),
          correlationId: this.generateCorrelationId(),
          payload: event
        };
        this.events.push(envelope);
      }
      
      for (const event of this.events) {
        this.notifySubscribers(event);
      }
    } catch (error) {
      console.error('Failed to load events from URL:', error);
    }
  }

  private stripDefaults(event: GameEvent): any {
    const stripped: any = {};
    for (const [key, value] of Object.entries(event)) {
      if (key !== 'type' && value !== undefined && value !== null && value !== false && value !== 0) {
        stripped[key] = value;
      }
    }
    return stripped;
  }

  private restoreDefaults(type: string, payload: any): GameEvent {
    return {
      type,
      ...payload
    } as GameEvent;
  }

  private chunkString(str: string, size: number): string[] {
    const chunks: string[] = [];
    for (let i = 0; i < str.length; i += size) {
      chunks.push(str.slice(i, i + size));
    }
    return chunks;
  }

  private lzCompress(data: Uint8Array): Uint8Array {
    const dict = new Map<string, number>();
    const result: number[] = [];
    let dictSize = 256;
    let w = '';
    
    for (const byte of data) {
      const c = String.fromCharCode(byte);
      const wc = w + c;
      
      if (dict.has(wc)) {
        w = wc;
      } else {
        const code = dict.has(w) ? dict.get(w)! : w.charCodeAt(0);
        result.push(code);
        
        if (dictSize < 4096) {
          dict.set(wc, dictSize++);
        }
        w = c;
      }
    }
    
    if (w) {
      const code = dict.has(w) ? dict.get(w)! : w.charCodeAt(0);
      result.push(code);
    }
    
    return new Uint8Array(result);
  }

  private lzDecompress(data: Uint8Array): Uint8Array {
    const dict = new Map<number, string>();
    const result: number[] = [];
    let dictSize = 256;
    
    for (let i = 0; i < 256; i++) {
      dict.set(i, String.fromCharCode(i));
    }
    
    let w = String.fromCharCode(data[0]);
    result.push(data[0]);
    
    for (let i = 1; i < data.length; i++) {
      const k = data[i];
      const entry = dict.has(k) ? dict.get(k)! : w + w[0];
      
      for (const char of entry) {
        result.push(char.charCodeAt(0));
      }
      
      if (dictSize < 4096) {
        dict.set(dictSize++, w + entry[0]);
      }
      
      w = entry;
    }
    
    return new Uint8Array(result);
  }
}