# kuramoto-bubble-detection
#Kuramoto synchronization model applied to financial bubble detection
Kuramoto Synchronization Model as a Financial Bubble Detector

## Where the idea came from

In a world where every object tends to follow its own individual behaviour, the emergence of synchronization between them is surprising. Metronomes swinging in the same rhythm, pendulum clocks aligning with each other, fireflies flashing in unison — these are all examples of the Kuramoto model in action. Studying this model, I began to wonder: does synchronization appear in financial markets too?

## How this connects to financial bubbles

As someone who has observed the rapid growth of AI startups and companies over the past five years, I found myself asking: could history repeat itself — this time as an AI bubble, echoing the dot-com crash of the early 2000s?
The core idea of the Kuramoto model is synchronization between objects. In financial markets, when AI companies cross a certain synchronization threshold — calculated using the Hilbert transform — investors fall into a herd effect: instead of making independent decisions, they begin to move in one rhythm. Think of an audience in a theatre: at first everyone claps at their own pace, but gradually people synchronize with the general rhythm of the room rather than with their neighbour. It is precisely this collective behaviour that leads to market overheating, the formation of a bubble, and eventually its collapse. Other contributing factors include hype, overestimation of technology, and speculative investment.

## About the data

The project contains two programs. kuramoto.py uses artificially generated data modelled to approximate real market behaviour — because historical data on dot-com era companies that went bankrupt is not publicly accessible. AI giants of that era are deliberately excluded, as they already had diversified revenue streams and did not follow the typical bubble pattern. kuramoto_real.py uses real data from major AI companies as the only available open source, serving as a reference point for comparison.
