// Helper to deep clone an object preserving Sets
export function deepClone<T>(obj: T): T {
  if (obj === null || obj === undefined) return obj;
  if (obj instanceof Set) return new Set(obj) as T;
  if (obj instanceof Date) return new Date(obj.getTime()) as T;
  if (obj instanceof Array) return obj.map(item => deepClone(item)) as T;
  if (typeof obj === 'object') {
    const cloned = {} as T;
    for (const key in obj) {
      if (Object.prototype.hasOwnProperty.call(obj, key)) {
        cloned[key] = deepClone(obj[key]);
      }
    }
    return cloned;
  }
  return obj;
}

// Helper to deep compare two objects
export function deepCompare(obj1: unknown, obj2: unknown, path: string = ''): string[] {
  const differences: string[] = [];
  
  if (obj1 === obj2) return differences;
  
  if (obj1 === null || obj2 === null || obj1 === undefined || obj2 === undefined) {
    differences.push(`${path}: ${JSON.stringify(obj1)} !== ${JSON.stringify(obj2)}`);
    return differences;
  }
  
  if (typeof obj1 !== typeof obj2) {
    differences.push(`${path}: type mismatch - ${typeof obj1} !== ${typeof obj2}`);
    return differences;
  }
  
  if (typeof obj1 !== 'object') {
    if (obj1 !== obj2) {
      differences.push(`${path}: ${JSON.stringify(obj1)} !== ${JSON.stringify(obj2)}`);
    }
    return differences;
  }
  
  if (Array.isArray(obj1) !== Array.isArray(obj2)) {
    differences.push(`${path}: array mismatch - one is array, other is not`);
    return differences;
  }
  
  if (Array.isArray(obj1)) {
    if (Array.isArray(obj2)) {
      if (obj1.length !== obj2.length) {
        differences.push(`${path}: array length mismatch - ${obj1.length} !== ${obj2.length}`);
      }
      const maxLen = Math.max(obj1.length, obj2.length);
      for (let i = 0; i < maxLen; i++) {
        differences.push(...deepCompare(obj1[i], obj2[i], `${path}[${i}]`));
      }
    }
  } else {
    const keys1 = Object.keys(obj1 as Record<string, unknown>).sort();
    const keys2 = Object.keys(obj2 as Record<string, unknown>).sort();
    
    // Check for missing/extra keys
    const allKeys = new Set([...keys1, ...keys2]);
    for (const key of allKeys) {
      if (!keys1.includes(key)) {
        differences.push(`${path}.${key}: missing in first object`);
      } else if (!keys2.includes(key)) {
        differences.push(`${path}.${key}: missing in second object`);
      } else {
        differences.push(...deepCompare((obj1 as Record<string, unknown>)[key], (obj2 as Record<string, unknown>)[key], path ? `${path}.${key}` : key));
      }
    }
  }
  
  return differences;
}