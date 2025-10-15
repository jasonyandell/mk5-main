<script lang="ts">
  import { gameVariants, oneHandState } from '../../stores/gameStore';
  import Icon from '../icons/Icon.svelte';

  const isActive = $derived($oneHandState.active);
  const isFindingSeed = $state(false); // TODO: Connect to seed finding progress

  function startOneHand() {
    gameVariants.startOneHand();
  }
</script>

<button
  class="btn btn-primary"
  onclick={startOneHand}
  disabled={isActive}
  data-testid="one-hand-button"
>
  {#if isFindingSeed}
    <span class="loading loading-spinner loading-sm"></span>
    Finding Hand...
  {:else if isActive}
    <Icon name="handRaised" size="sm" />
    Playing One Hand
  {:else}
    <Icon name="handRaised" size="sm" />
    Play One Hand
  {/if}
</button>