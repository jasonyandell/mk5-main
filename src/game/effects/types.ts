export interface Effect {
  id: string;
  type: 'ai-action' | 'animation' | 'sound';
  data: any;
  executeAt: number;
}

export type QuickplaySpeed = 'instant' | 'fast' | 'normal';

export interface QuickplaySettings {
  enabled: boolean;
  speed: QuickplaySpeed;
}

export interface ScheduledEffect {
  effect: Effect;
  timeoutId?: number;
}