import numpy as np

dataPath = '../datasets/'

class MinibooneLoader():
    """
    Miniboone dataloader. Stores the total count per-class and all loaded events
    
    Attributes
    ----------
    totalEvents (int) : total events loaded from the dataset.
    totalSignals (int) : total signal events loaded from the dataset.
    totalBackgrounds (int) : total backgrounds loaded from the dataset.

    Methods
    -------
    loadMiniboone()
        Loads and initialises itself from the dataset.
    linesAsNums()
        Iterator for a given file buffer.

    """

    totalEvents = 0
    totalSignals = 0
    totalBackgrounds = 0

    events = []
    classifications = []

    def loadMiniboone(self):
        """
        Loads and initialises itself from the dataset.

        Returns:
            self (MinibooneLoader): It's own self for fluent invocation.
        """
        eventsBuffer = []

        with open(f'{dataPath}MiniBooNE_PID.txt', 'r') as dataFile:

            # Loading and parsing the event totals.
            totals = dataFile.readline().strip()
            self.totalSignals, self.totalBackgrounds = [int(i) for i in totals.split(' ')]
            self.totalEvents = self.totalBackgrounds + self.totalSignals

            # Parse the data points in the file to a raw array.
            eventsBuffer = [event for event in self.linesAsNums(dataFile)]
    
        # Map the data to a new numpy array.
        self.events = np.array(eventsBuffer)
        
        signals = np.full(self.totalSignals, 0)
        backgrounds = np.full(self.totalBackgrounds, 1)

        self.classifications = np.concatenate((signals, backgrounds))
        return self
    
    def linesAsNums(self, file):
        """
        Iterator for a given file buffer, performing the following:
        * Stripping whitespace
        * Splitting by whitespace.
        * Parsing each substring as a float.
        * Yields it.

        Args:
            file ([IO]): A valid file buffer, starting at the first line to load.

        Yields:
            floats (float[]): The obtained floats from the file.
        """
        strings = []
        for line in file:
            strings = line.strip().split()
            yield [float(num) for num in strings]
