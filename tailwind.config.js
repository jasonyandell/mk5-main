/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{svelte,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      animation: {
        'score-bounce': 'scoreBounce 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
        'phase-in': 'phaseIn 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        'subtle-pulse': 'subtlePulse 2s infinite',
        'shake': 'shake 0.5s',
        'fadeInUp': 'fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        'fadeInDown': 'fadeInDown 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        'handFadeIn': 'handFadeIn 0.4s cubic-bezier(0.4, 0, 0.2, 1) both',
      },
      keyframes: {
        scoreBounce: {
          '0%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.3)' },
          '100%': { transform: 'scale(1)' }
        },
        phaseIn: {
          'from': { opacity: '0', transform: 'translateX(-20px)' },
          'to': { opacity: '1', transform: 'translateX(0)' }
        },
        subtlePulse: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' }
        },
        shake: {
          '0%, 100%': { transform: 'translateX(0)' },
          '25%': { transform: 'translateX(-5px)' },
          '75%': { transform: 'translateX(5px)' }
        },
        fadeInUp: {
          'from': { opacity: '0', transform: 'translateY(20px)' },
          'to': { opacity: '1', transform: 'translateY(0)' }
        },
        fadeInDown: {
          'from': { opacity: '0', transform: 'translateY(-10px)' },
          'to': { opacity: '1', transform: 'translateY(0)' }
        },
        handFadeIn: {
          'from': { opacity: '0', transform: 'translateY(10px) rotate(-5deg) scale(0.9)' },
          'to': { opacity: '1', transform: 'translateY(0) rotate(0) scale(1)' }
        }
      },
      colors: {
        'game-bg': {
          light: '#f8fafc',
          DEFAULT: '#e2e8f0'
        }
      },
      spacing: {
        'safe': 'env(safe-area-inset-bottom)',
        'safe-top': 'env(safe-area-inset-top)',
      },
      minHeight: {
        'touch': '44px',
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [require("daisyui")],
  daisyui: {
    themes: ["light", "dark", "cupcake", "retro", "cyberpunk", "valentine", "forest", "luxury", "dracula", "coffee"],
    darkTheme: "dark",
    base: true,
    styled: true,
    utils: true,
    prefix: "",
    logs: true,
    themeRoot: ":root",
  },
}