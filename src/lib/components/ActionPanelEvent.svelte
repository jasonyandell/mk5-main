<script lang="ts">
  import { eventGame } from '../../stores/eventGame';
  import type { Bid, TrumpSelection } from '../../game/types';

  const gamePhase = eventGame.gamePhase;
  const currentPlayer = eventGame.currentPlayer;
  const bids = eventGame.bids;

  function handleBid(type: Bid['type'], value?: number) {
    const bid: Bid = { type, value, player: $currentPlayer };
    eventGame.placeBid($currentPlayer, bid);
  }

  function handlePass() {
    eventGame.pass($currentPlayer);
  }

  function handleTrump(trump: TrumpSelection) {
    eventGame.selectTrump($currentPlayer, trump);
  }

  $: isBiddingPhase = $gamePhase === 'bidding';
  $: isTrumpPhase = $gamePhase === 'trump_selection';
</script>

<div class="card bg-base-100 shadow-xl h-full">
  <div class="card-body">
    <h2 class="card-title">Actions</h2>
    
    {#if isBiddingPhase}
      <div class="space-y-4">
        <h3 class="text-lg font-semibold">Bidding Options</h3>
        
        <div class="grid grid-cols-2 gap-2">
          <button class="btn btn-primary" onclick={() => handleBid('points', 30)}>
            30 Points
          </button>
          <button class="btn btn-primary" onclick={() => handleBid('points', 31)}>
            31 Points
          </button>
          <button class="btn btn-primary" onclick={() => handleBid('points', 32)}>
            32 Points
          </button>
          <button class="btn btn-primary" onclick={() => handleBid('points', 33)}>
            33 Points
          </button>
          <button class="btn btn-primary" onclick={() => handleBid('points', 34)}>
            34 Points
          </button>
          <button class="btn btn-primary" onclick={() => handleBid('points', 35)}>
            35 Points
          </button>
        </div>
        
        <div class="grid grid-cols-2 gap-2">
          <button class="btn btn-secondary" onclick={() => handleBid('marks', 1)}>
            1 Mark
          </button>
          <button class="btn btn-secondary" onclick={() => handleBid('marks', 2)}>
            2 Marks
          </button>
        </div>
        
        <button class="btn btn-outline w-full" onclick={handlePass}>
          Pass
        </button>
      </div>
    {/if}
    
    {#if isTrumpPhase}
      <div class="space-y-4">
        <h3 class="text-lg font-semibold">Select Trump</h3>
        
        <div class="grid grid-cols-2 gap-2">
          <button class="btn btn-primary" onclick={() => handleTrump({ type: 'suit', suit: 0 })}>
            Blanks
          </button>
          <button class="btn btn-primary" onclick={() => handleTrump({ type: 'suit', suit: 1 })}>
            Ones
          </button>
          <button class="btn btn-primary" onclick={() => handleTrump({ type: 'suit', suit: 2 })}>
            Twos
          </button>
          <button class="btn btn-primary" onclick={() => handleTrump({ type: 'suit', suit: 3 })}>
            Threes
          </button>
          <button class="btn btn-primary" onclick={() => handleTrump({ type: 'suit', suit: 4 })}>
            Fours
          </button>
          <button class="btn btn-primary" onclick={() => handleTrump({ type: 'suit', suit: 5 })}>
            Fives
          </button>
          <button class="btn btn-primary" onclick={() => handleTrump({ type: 'suit', suit: 6 })}>
            Sixes
          </button>
          <button class="btn btn-secondary" onclick={() => handleTrump({ type: 'doubles' })}>
            Doubles
          </button>
        </div>
        
        <button class="btn btn-outline w-full" onclick={() => handleTrump({ type: 'no-trump' })}>
          No Trump
        </button>
      </div>
    {/if}
    
    <div class="mt-4">
      <div class="text-sm opacity-70">
        <p>Current Player: {$currentPlayer + 1}</p>
        <p>Phase: {$gamePhase}</p>
        {#if $bids.highestBid}
          <p>Highest Bid: {$bids.highestBid.type} {$bids.highestBid.value || ''}</p>
        {/if}
      </div>
    </div>
  </div>
</div>