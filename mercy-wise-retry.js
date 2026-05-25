// mercy-wise-retry.js
// Ra-Thor Lattice — Wise & Efficient Retry Strategy
// Prevents system and downstream overloading through exponential backoff with jitter,
// circuit breaker pattern, smart transient vs permanent error classification,
// and per-attempt timeouts.
// Designed for orchestration, swarm coordination, API calls, and any resilient operation.
// Fully compatible with existing mercy-orchestrator.js and swarm modules.
// AG-SML v1.0 (Autonomicity Games Sovereign Mercy License)
// Version: 1.0.1 | Date: 2026-05-24

/**
 * MercyWiseRetry
 * 
 * Usage:
 *   import { MercyWiseRetry } from './mercy-wise-retry.js';
 *   
 *   const retry = new MercyWiseRetry({
 *     maxRetries: 5,
 *     baseDelay: 150,
 *     maxDelay: 30000,
 *     timeout: 8000,
 *     circuitBreakerThreshold: 3,
 *     circuitBreakerCooldown: 60000
 *   });
 * 
 *   const result = await retry.execute(async () => {
 *     const res = await fetch('https://api.example.com/data');
 *     if (!res.ok) throw { status: res.status, message: res.statusText };
 *     return res.json();
 *   });
 */
export class MercyWiseRetry {
  constructor(options = {}) {
    this.maxRetries = options.maxRetries ?? 5;
    this.baseDelay = options.baseDelay ?? 100; // milliseconds
    this.maxDelay = options.maxDelay ?? 30000;
    this.jitter = options.jitter !== false; // full jitter enabled by default for load distribution
    this.timeout = options.timeout ?? 10000;
    this.circuitBreakerThreshold = options.circuitBreakerThreshold ?? 3;
    this.circuitBreakerCooldown = options.circuitBreakerCooldown ?? 60000; // 1 minute

    // Internal state
    this.failureCount = 0;
    this.circuitOpenUntil = 0;
    this.lastError = null;
  }

  /**
   * Classify whether an error is worth retrying.
   * Avoids retrying on client errors (4xx) that would waste resources or indicate permanent issues.
   * Retries on server errors (5xx), rate limits (429), and common transient network problems.
   */
  isRetryableError(error) {
    if (!error) return false;

    const status = error.status || error.code || (error.response && error.response.status);
    const message = (error.message || '').toLowerCase();

    // Explicit rate limit — always retry (server is asking us to back off)
    if (status === 429) return true;

    // Server-side errors are transient and worth retrying
    if (status >= 500 && status < 600) return true;

    // Client errors (4xx) are usually permanent — do not retry to avoid overloading
    // the target or wasting cycles on bad requests
    if (status >= 400 && status < 500) return false;

    // Common transient network / timeout conditions
    if (
      error.code === 'ECONNRESET' ||
      error.code === 'ETIMEDOUT' ||
      error.code === 'ECONNREFUSED' ||
      message.includes('network') ||
      message.includes('fetch failed') ||
      message.includes('timeout')
    ) {
      return true;
    }

    // Default conservative: do not retry unknown errors
    return false;
  }

  /**
   * Calculate next delay using exponential backoff + full jitter.
   * Full jitter spreads retries randomly to prevent thundering herd / overloading services.
   */
  calculateDelay(attempt) {
    const exponential = Math.min(
      this.baseDelay * Math.pow(2, attempt),
      this.maxDelay
    );

    if (!this.jitter) {
      return exponential;
    }

    // Full jitter: random value between 0 and calculated exponential
    return Math.floor(Math.random() * exponential);
  }

  /**
   * Wrap a promise with a timeout.
   */
  async _withTimeout(promise, ms) {
    let timeoutId;
    const timeoutPromise = new Promise((_, reject) => {
      timeoutId = setTimeout(() => {
        reject(new Error(`Operation timed out after ${ms}ms`));
      }, ms);
    });

    try {
      return await Promise.race([promise, timeoutPromise]);
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Record a failure and potentially open the circuit breaker.
   */
  _recordFailure() {
    this.failureCount += 1;
    if (this.failureCount >= this.circuitBreakerThreshold) {
      this.circuitOpenUntil = Date.now() + this.circuitBreakerCooldown;
      console.warn(
        `[MercyWiseRetry] Circuit breaker OPENED — cooling down for ${this.circuitBreakerCooldown}ms to protect against overload. Consecutive failures: ${this.failureCount}`
      );
    }
  }

  /**
   * Reset internal state after successful operation or manual intervention.
   */
  reset() {
    this.failureCount = 0;
    this.circuitOpenUntil = 0;
    this.lastError = null;
  }

  /**
   * Execute an async function with intelligent retry, backoff, jitter, timeout, and circuit breaking.
   * 
   * @param {Function} fn - Async function to execute (must return a Promise)
   * @param {Object} [context] - Optional context for logging (e.g. { operation: 'fetchUser' })
   * @returns {Promise<any>} Result of successful execution
   */
  async execute(fn, context = {}) {
    const operationName = context.operation || 'unknown-operation';

    // Circuit breaker check
    if (Date.now() < this.circuitOpenUntil) {
      const remaining = Math.ceil((this.circuitOpenUntil - Date.now()) / 1000);
      throw new Error(
        `[MercyWiseRetry] Circuit breaker is OPEN for ${remaining}s. ` +
        `Too many failures on "${operationName}". System protecting itself from overload.`
      );
    }

    let lastError = null;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const result = await this._withTimeout(fn(), this.timeout);

        // Success path — reset state
        if (this.failureCount > 0) {
          console.log(`[MercyWiseRetry] "${operationName}" recovered after ${this.failureCount} failures.`);
        }
        this.reset();
        return result;

      } catch (error) {
        lastError = error;
        this.lastError = error;

        const isLastAttempt = attempt === this.maxRetries;
        const shouldRetry = this.isRetryableError(error) && !isLastAttempt;

        if (!shouldRetry) {
          this._recordFailure();
          // Re-throw with context
          const wrappedError = new Error(
            `[MercyWiseRetry] "${operationName}" failed permanently after ${attempt + 1} attempt(s). ` +
            `Last error: ${error.message || error}`
          );
          wrappedError.originalError = error;
          wrappedError.attempts = attempt + 1;
          throw wrappedError;
        }

        // Calculate smart delay
        const delay = this.calculateDelay(attempt);
        console.log(
          `[MercyWiseRetry] "${operationName}" attempt ${attempt + 1}/${this.maxRetries + 1} failed. ` +
          `Retrying in ${delay}ms (backoff + jitter). Reason: ${error.message || error}`
        );

        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    // Should not reach here, but safety
    this._recordFailure();
    throw lastError;
  }
}

// Default singleton instance for convenience in monorepo modules
export const mercyWiseRetry = new MercyWiseRetry();

// Quick helper for one-off usage
export async function withMercyRetry(fn, options = {}, context = {}) {
  const retry = new MercyWiseRetry(options);
  return retry.execute(fn, context);
}

export default MercyWiseRetry;