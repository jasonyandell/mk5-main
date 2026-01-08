# Reddit Games / Devvit Development Skill

You are now a Reddit Games (Devvit) development expert. Use this comprehensive guide to help build interactive apps and games for Reddit.

## What is Devvit?

Devvit is Reddit's Developer Platform for building interactive games and apps that live on Reddit. Apps can earn up to $167,000 through Reddit Developer Funds.

## Quick Start

### Environment Setup
1. Install Node.js 22.2.0+
2. Go to https://developers.reddit.com/new
3. Choose a template (React, Unity, Three.js, Phaser, or Hello World)
4. Run `npm run dev` to start development

### Project Structure
```
your-app/
├── src/
│   ├── client/       # Web app code (React, Vue, etc.)
│   ├── server/       # Node.js server (Express, etc.)
│   └── shared/       # Shared types/interfaces
├── devvit.json       # App configuration
└── package.json
```

## Architecture Patterns

### Devvit Web (Recommended)
- Standard web stack with React, Three.js, Phaser, etc.
- Client/server split architecture
- Server endpoints via Express.js or similar
- Import from `@devvit/web/client` and `@devvit/web/server`

### Devvit Blocks (Legacy)
- Reddit's component system
- Uses `Devvit.addCustomPostType()` and JSX-like syntax
- Import from `@devvit/public-api`

## devvit.json Configuration

```json
{
  "$schema": "https://developers.reddit.com/schema/config-file.v1.json",
  "name": "my-app-name",
  "post": {
    "dir": "dist/client",
    "entrypoints": {
      "default": {
        "entry": "index.html",
        "height": "tall"
      }
    }
  },
  "server": {
    "entry": "dist/server/index.cjs"
  },
  "permissions": {
    "redis": true,
    "reddit": true,
    "http": {
      "enable": true,
      "domains": ["api.example.com"]
    }
  },
  "triggers": {
    "onPostCreate": "/internal/triggers/post-create"
  },
  "menu": {
    "items": [{
      "label": "Create Post",
      "location": "subreddit",
      "forUserType": "moderator",
      "endpoint": "/internal/menu/create-post"
    }]
  },
  "scheduler": {
    "tasks": {
      "daily-task": {
        "endpoint": "/internal/cron/daily",
        "cron": "0 2 * * *"
      }
    }
  }
}
```

## Server Capabilities

### Redis Storage
```typescript
import { redis } from '@devvit/web/server';

await redis.set('key', 'value');
const value = await redis.get('key');
await redis.hSet('hash', { field: 'value' });
await redis.zAdd('leaderboard', { member: 'user1', score: 100 });
```

### Reddit API
```typescript
import { reddit, context } from '@devvit/web/server';

const post = await reddit.submitCustomPost({
  subredditName: context.subredditName!,
  title: 'My Post',
  entry: 'default',
  postData: { gameState: 'initial' }
});
```

### Scheduler
```typescript
import { scheduler } from '@devvit/web/server';

await scheduler.runJob({
  name: 'my-task',
  runAt: new Date(Date.now() + 60000),
  data: { postId: 'abc123' }
});
```

### Settings
```typescript
import { settings } from '@devvit/web/server';

const apiKey = await settings.get('apiKey');
```

## Client Effects

```typescript
import { showToast, showForm, navigateTo, purchase } from '@devvit/web/client';

// Toast notifications
showToast('Success!');
showToast({ text: 'Saved!', appearance: 'success' });

// Forms
const result = await showForm({
  form: {
    fields: [
      { type: 'string', name: 'username', label: 'Username' }
    ]
  }
});

// Navigation
navigateTo('https://www.reddit.com/r/gaming');

// Payments
const result = await purchase('product-sku');
```

## Realtime

### Client
```typescript
import { connectRealtime } from '@devvit/web/client';

const connection = await connectRealtime({
  channel: 'game-updates',
  onMessage: (data) => console.log('Received:', data),
  onConnect: () => console.log('Connected'),
  onDisconnect: () => console.log('Disconnected')
});

await connection.disconnect();
```

### Server
```typescript
import { realtime } from '@devvit/web/server';

await realtime.send('game-updates', { type: 'score', value: 100 });
```

## View Modes & Entry Points

### Inline Mode (Default)
- Loads in post unit
- Must only use tap/click
- Load in under 1 second
- No scroll hijacking

### Expanded Mode
- Full screen on mobile, modal on web
- User-initiated only

```typescript
import { requestExpandedMode, exitExpandedMode, getWebViewMode } from '@devvit/web/client';

// Enter expanded mode
await requestExpandedMode(event.nativeEvent, 'game');

// Exit expanded mode
await exitExpandedMode(event.nativeEvent);

// Check current mode
const mode = getWebViewMode(); // 'inline' | 'expanded'
```

## Payments

### Setup products.json
```json
{
  "$schema": "https://developers.reddit.com/schema/products.json",
  "products": [{
    "sku": "power-up",
    "displayName": "Power Up",
    "description": "Gives you extra abilities",
    "price": 50,
    "accountingType": "CONSUMABLE"
  }]
}
```

### Server Handler
```typescript
router.post('/internal/payments/fulfill', async (req, res) => {
  const { products, userId } = req.body;
  // Grant product to user
  res.json({ success: true });
});
```

### Client Purchase
```typescript
import { purchase, OrderResultStatus } from '@devvit/web/client';

const result = await purchase('power-up');
if (result.status === OrderResultStatus.STATUS_SUCCESS) {
  // Handle success
}
```

## Triggers

Available triggers:
- `onAppInstall`, `onAppUpgrade`
- `onPostCreate`, `onPostDelete`, `onPostSubmit`, `onPostUpdate`, `onPostReport`
- `onCommentCreate`, `onCommentDelete`, `onCommentSubmit`, `onCommentUpdate`
- `onModAction`, `onModMail`

```typescript
router.post('/internal/triggers/post-create', async (req, res) => {
  const { post, author } = req.body;
  console.log(`New post by ${author.name}`);
  res.json({ status: 'ok' });
});
```

## Menu Actions

```typescript
router.post('/internal/menu/my-action', async (req, res) => {
  res.json({
    showToast: { text: 'Action completed!', appearance: 'success' }
  });
});
```

## Best Practices

### Game Design
1. Keep gameplay bite-sized (under 2 minutes)
2. Design for the feed (eye-catching first screen)
3. Build content flywheels (daily challenges, user-generated content)
4. Embrace asynchronous play
5. Scale from 1 to many players

### Performance
- Optimize for mobile
- Load initial content in under 1 second
- Target Lighthouse score >80
- Use Redis caching for expensive operations

### Engagement
- Add leaderboards
- Include subscribe buttons
- Use community flair
- Enable user-generated content

## CLI Commands

```bash
devvit login              # Authenticate
devvit new [name]         # Create new app
devvit playtest [sub]     # Test on subreddit
devvit upload             # Upload to directory
devvit publish            # Publish app
devvit logs <subreddit>   # Stream logs
devvit install <sub>      # Install app
```

## Templates

- **React**: https://github.com/reddit/devvit-template-react
- **Three.js**: https://github.com/reddit/devvit-template-threejs
- **Phaser**: https://github.com/reddit/devvit-template-phaser
- **Unity**: Via wizard at developers.reddit.com/new

## Resources

- Docs: https://developers.reddit.com/docs
- Discord: https://discord.gg/Cd43ExtEFS
- Community: r/devvit
- App Showcase: r/GamesOnReddit

## Developer Funds

Earn up to $167,000 per app through Reddit Developer Funds:
- 500 daily qualified engagers: $500
- 1,000 daily qualified engagers: $1,500 cumulative
- 10,000 daily qualified engagers: $6,500 cumulative
- Up to 1,000,000+ daily qualified engagers: $167,000 cumulative

Requirements:
- Subreddit with 200+ members
- Safe for Work content
- Logged-in user engagement
