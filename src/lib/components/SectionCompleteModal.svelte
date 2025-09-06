<script lang="ts">
  import { sectionOverlay, sectionActions } from '../../stores/gameStore';
  import Icon from '../icons/Icon.svelte';

  let copySuccess = $state(false);

  import { initialState } from '../../stores/gameStore';
  import { buildUrl } from '../../stores/utils/urlManager';
  const newRun = () => {
    if (typeof window !== 'undefined') {
      const url = buildUrl({
        initialState: $initialState,
        actionIds: [],
        scenarioName: 'one_hand',
        includeSeed: false,
        includeActions: false,
        includeScenario: true,
        includeTheme: true,
        includeOverrides: false,
        preserveUnknownParams: true,
        absolute: true
      });
      window.location.assign(url);
    }
  };
  const retry = () => {
    const seed = $sectionOverlay?.seed;
    sectionActions.restartOneHand(seed);
  };
  const challenge = async () => {
    const seed = $sectionOverlay?.seed;
    const tries = $sectionOverlay?.attemptsCount ?? 1;
    if (typeof window !== 'undefined' && seed) {
      // Build encoded challenge URL with explicit seed and scenario
      const baseInitial = { ...$initialState, shuffleSeed: seed } as typeof $initialState;
      const url = buildUrl({
        initialState: baseInitial,
        actionIds: [],
        scenarioName: 'one_hand',
        includeSeed: true,
        includeActions: false,
        includeScenario: true,
        includeTheme: true,
        includeOverrides: false,
        preserveUnknownParams: true,
        absolute: true
      });
      const challengeText = tries === 1 
        ? "42 🁠 Skunked 'em on the first try!" 
        : `42 🁠 Got it in ${tries} tries! Can you beat that?`;
      const text = `${challengeText}\n${url}`;
      try {
        await navigator.clipboard.writeText(text);
        copySuccess = true;
        setTimeout(() => copySuccess = false, 2000);
      } catch (e) {
        console.warn('Clipboard write failed', e);
      }
    }
  };
</script>

{#if $sectionOverlay && $sectionOverlay.type === 'oneHand'}
  <div class="modal modal-open">
    <div class="modal-box max-w-[12rem] w-full">
      <div class="text-center">
        <h3 class="font-bold text-2xl mb-2">
          {$sectionOverlay.weWon ? 'We Won!' : 'We Lost'}
        </h3>
        <p class="text-base-content/70 mb-4">{$sectionOverlay.weWon ? 'Of course.' : "We'll get 'em next time, partner"}</p>
        <div class="text-lg font-semibold mb-6">
          <span class="font-bold">US</span> {$sectionOverlay.usScore ?? 0} • <span class="font-bold">THEM</span> {$sectionOverlay.themScore ?? 0}
        </div>
        {#if $sectionOverlay.attemptsForWin && $sectionOverlay.canChallenge}
          <p class="text-sm opacity-75 mb-6">You beat this seed in {$sectionOverlay.attemptsForWin} {$sectionOverlay.attemptsForWin === 1 ? 'try' : 'tries'}.</p>
        {/if}
      </div>
      <div class="flex flex-col gap-2">
        <button class="btn btn-primary w-full" onclick={newRun}>New Game</button>
        {#if !$sectionOverlay.weWon}
          <button class="btn btn-accent w-full" onclick={retry}>Retry ({$sectionOverlay.attemptsCount ?? 1})</button>
        {/if}
        {#if $sectionOverlay.canChallenge}
          <button class="btn {copySuccess ? 'btn-success' : 'btn-secondary'} w-full" onclick={challenge}>
            {#if copySuccess}
              <Icon name="check" size="sm" />
              Copied!
            {:else}
              <Icon name="link" size="sm" />
              Challenge!
            {/if}
          </button>
        {/if}
      </div>
    </div>
  </div>
{/if}
