import { title, overview, pipeline, stft, mdx, bandit, detect } from './scenes1.js';
import { track, siglip, decide, render, runtime, results, outro } from './scenes2.js';

export const MAKERS = { title, overview, pipeline, stft, mdx, bandit, detect, track, siglip, decide, render, runtime, results, outro };

export const CHAPTERS = {
  title: 'HalalifyVid',
  overview: 'What it is',
  pipeline: 'The pipeline',
  stft: 'Sound into a spectrogram',
  mdx: 'MDX-Net: keep the voices',
  bandit: 'BandIt Plus: keep the speech',
  detect: 'Find every person',
  track: 'Follow each person',
  siglip: 'Who is a woman?',
  decide: 'Two passes, one decision',
  render: 'Render with masks',
  runtime: 'Runtime and setup',
  results: 'Speed and accuracy',
  outro: 'Use it',
};
