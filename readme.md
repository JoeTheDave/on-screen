# On-Screen

A web app that prevents your computer from going to sleep and displays canvas-based animations.

## Deployment to Fly.io

### First-time setup

1. Install the Fly CLI: https://fly.io/docs/hands-on/install-flyctl/
2. Login to Fly: `fly auth login`
3. Launch the app: `fly launch --no-deploy`
4. Deploy: `fly deploy`

### Subsequent deployments

```bash
fly deploy
```

production url: `https://on-screen.fly.dev` 


