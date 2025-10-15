<script lang="ts">
  import { gameVariants, oneHandState } from '../../stores/gameStore';
  import Icon from '../icons/Icon.svelte';
  import { shareContent, canNativeShare } from '../utils/share';

  const oneHandInfo = $derived($oneHandState);
  const show = $derived(oneHandInfo.complete);
  const result = $derived(oneHandInfo.scores);

  let shareSuccess = $state(false);

  function retry() {
    gameVariants.retryOneHand();
  }

  function newHand() {
    gameVariants.startOneHand();
  }

  function exitOneHand() {
    gameVariants.exitVariant();
  }

  async function shareChallenge() {
    if (!result || !oneHandInfo.seed) return;

    const baseUrl = window.location.origin + window.location.pathname;
    const challengeUrl = `${baseUrl}?onehand=${oneHandInfo.seed}`;

    const challengeText = oneHandInfo.attempts === 1
      ? "42 🁠 Can you beat this hand?"
      : `42 🁠 Got it in ${oneHandInfo.attempts} tries! Can you beat that?`;

    const success = await shareContent({
      title: '42 🁠 Challenge',
      text: challengeText,
      url: challengeUrl
    });

    if (success) {
      shareSuccess = true;
      setTimeout(() => shareSuccess = false, 2000);
    }
  }
</script>

{#if show && result}
  <div class="modal modal-open">
    <div class="modal-box max-w-xs">
      <div class="text-center">
        <h3 class="font-bold text-2xl mb-2">
          {result.won ? 'Victory!' : 'Defeat'}
        </h3>
        <p class="text-base-content/70 mb-4">
          {result.won ? 'Well played, partner!' : "We'll get 'em next time!"}
        </p>
        <div class="text-lg font-semibold mb-6">
          <span class="font-bold">US</span> {result.us} • <span class="font-bold">THEM</span> {result.them}
        </div>
        {#if oneHandInfo.attempts > 1}
          <p class="text-sm opacity-75 mb-4">
            Attempt #{oneHandInfo.attempts}
          </p>
        {/if}
      </div>

      <div class="flex flex-col gap-2">
        {#if !result.won}
          <button class="btn btn-accent w-full" onclick={retry}>
            Retry Same Hand
            {#if oneHandInfo.attempts > 0}
              (#{oneHandInfo.attempts + 1})
            {/if}
          </button>
        {/if}

        <button class="btn btn-primary w-full" onclick={newHand}>
          New Hand
        </button>

        <button
          class="btn {shareSuccess ? 'btn-success' : 'btn-secondary'} w-full"
          onclick={shareChallenge}
        >
          {#if shareSuccess}
            <Icon name="check" size="sm" />
            {canNativeShare() ? 'Shared!' : 'Link Copied!'}
          {:else}
            <Icon name={canNativeShare() ? 'share' : 'link'} size="sm" />
            Share Challenge
          {/if}
        </button>

        <button class="btn btn-ghost w-full" onclick={exitOneHand}>
          Exit One Hand Mode
        </button>
      </div>
    </div>
  </div>
{/if}