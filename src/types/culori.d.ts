declare module 'culori' {
  export interface Color {
    mode: string;
    l?: number;
    c?: number;
    h?: number;
    s?: number;
    r?: number;
    g?: number;
    b?: number;
    alpha?: number;
  }

  export function formatHex(color: Color): string | undefined;
  export function parse(color: string): Color | undefined;
  export function converter(mode: string): (color: Color) => Color | undefined;
}