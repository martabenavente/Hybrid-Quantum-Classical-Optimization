import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt

def generateSetup(L, K, N, tau_p, ASD_varphi, numActiveAPs, grid=True, semilla=False):


    if semilla:
        np.random.seed(semilla)

    # Simulation Setup Configuration Parameters
    squarelength = 1000         # length of one side the coverage area in m (assuming wrap-around)

    B = 20*10**6                # communication bandwidth in Hz
    noiseFigure = 7             # noise figure in dB
    noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm

    alpha = 36.7                # pathloss parameters for the path loss model
    constantTerm = -30.5

    sigma_sf = 4                # standard deviation of the shadow fading
    decorr = 9                  # decorrelatiojn distance of the shadow fading

    distanceVertical = 10       # height difference between the APs and the UEs in meters
    antennaSpacing = 0.5        # half-wavelength distance

    # To save the results
    gainOverNoisedB = np.zeros((L, K))
    powgain = np.zeros((L, K))
    R = np.zeros((N, N, L, K), dtype=complex)
    distances = np.zeros((L, K))
    pilotIndex = np.zeros((K))
    D = np.zeros((L, K))
    masterAPs = np.zeros((K, 1))        # stores the indices of the master AP of every UE
    APpositions = np.zeros((L, 1))
    UEpositions = np.zeros((K, 1), dtype=complex)

    if grid:
        # Número de puntos por lado (aproximar la raíz cuadrada de L)
        num_points_side = int(np.floor(np.sqrt(L)))

        # Espaciado entre puntos
        spacing = squarelength / (num_points_side)

        # Generar coordenadas de la rejilla
        x = np.arange(spacing / 2, squarelength, spacing)
        y = np.arange(spacing / 2, squarelength, spacing)

        # Crear la rejilla cartesiana
        X, Y = np.meshgrid(x, y)
        APpositions = (X.flatten() + 1j * Y.flatten()).reshape(-1, 1)

        extra = L-len(APpositions)
        if extra > 0:
            # Calcular puntos medios entre APs
            mid_x = (X[:-1, :-1] + X[1:, 1:]) / 2
            mid_y = (Y[:-1, :-1] + Y[1:, 1:]) / 2

            # Añadir las posiciones centrales hasta alcanzar L APs
            APpositions_extra = (mid_x.flatten()[:extra] + 1j * mid_y.flatten()[:extra]).reshape(-1, 1)
            APpositions = np.vstack((APpositions, APpositions_extra))

        # print(f'APs ordenadas: {APpositions}')
    else:
        APpositions = (np.random.rand(L,1) + 1j*np.random.rand(L,1))*squarelength     # random AP locations with uniform distribution

    active_APs = np.zeros((K, numActiveAPs), dtype=int)


    # To save the shadowing correlation matrices
    shadowCorrMatrix = sigma_sf**2*np.ones((K, K))
    shadowAPrealizations = np.zeros((K, L))

    # Add UEs
    for k in range(K):
        # generate a random UE location with uniform distribution
        UEposition = (np.random.rand() + 1j*np.random.rand())*squarelength

        # compute distance from new UE to all the APs
        distances[:, k] = np.sqrt(distanceVertical**2+np.abs(APpositions-UEposition)**2)[:,0]


        # Select the 10 closest APs
        closest_APs = np.argsort(distances[:, k])[:numActiveAPs]  # Indices
        # print(f'Closest APs: {closest_APs}')
        active_APs[k, :] = closest_APs
        # print(f'Active APs for {k}: {active_APs}')
        # print(f'Active APs [0][0]: {active_APs[0][0]}')



        shadowing = np.zeros(L)
        gainOverNoisedB[:, k] = 0
        powgain[:, k] = 0

        if k > 0:         # if UE k is not the first UE
            shortestDistances = np.zeros((k,1))

            for i in range(k):
                shortestDistances[i] = min(np.abs(UEposition-UEpositions[i]))

            # Compute conditional mean and standard deviation necessary to obtain the new shadow fading
            # realizations when the previous UEs' shadow fading realization have already been generated
            newcolumn = (sigma_sf**2)*(2**(shortestDistances/-(decorr)))[:,0]
            term1 = newcolumn.conjugate().T@alg.inv(shadowCorrMatrix[:k, :k])
            meanvalues = term1@shadowAPrealizations[:k,:]
            stdvalue = np.sqrt(sigma_sf**2 - term1@newcolumn)

            shadowing = meanvalues + stdvalue*np.random.randn(L)

        else:           # if UE k is the first UE
            meanvalues = [0]*L
            stdvalue = sigma_sf
            newcolumn = np.array([])

            shadowing = meanvalues + stdvalue*np.random.randn(L)

        # print(f'El shadowing para el usuario {k} es: {shadowing}')


        for ap in range(L):
            # Compute the channel gain divided by noise power
            gainOverNoisedB[ap, k] = constantTerm - alpha*np.log10(distances[ap, k]) + shadowing[ap] - noiseVariancedBm
            if gainOverNoisedB[ap, k] != 0:
                powgain[ap, k] = db2pow(gainOverNoisedB[ap, k])
        # print(f'gainOverNoisedb para el usuario {k}: {gainOverNoisedB}')
        # print(f'Shadowing: {shadowing}')

        # Update shadowing correlation matrix and store realizations
        shadowCorrMatrix[0:k, k] = newcolumn
        shadowCorrMatrix[k, 0:k] = newcolumn.T
        shadowAPrealizations[k, :] = shadowing

        # print(f'shadowcorrmatrix para {k}: {shadowCorrMatrix}')
        # print(f'shadowAPrealizations para {k}: {shadowAPrealizations}')

        # store the UE position
        UEpositions[k] = UEposition

        # Determine the master AP for UE k by looking for the AP with best channel condition
        valid = np.where(gainOverNoisedB[:, k] != 0)[0]
        master = np.argmax(gainOverNoisedB[valid, k])
        D[master, k] = 1
        masterAPs[k] = master

        # print(f'APs para {k}: {closest_APs}')
        # print(f'master for k {k}: {master}')
        # print(f'masterAPs: {masterAPs}')
        # print(f'D para {k}: {D}')

        if k <= tau_p-1:        # Assign orthogonal pilots to the first tau_p UEs
            pilotIndex[k] = k

        else:                   # Assign pilot for remaining users

            # Compute received power to the master AP from each pilot
            pilotInterference = np.zeros(tau_p)

            for t in range(tau_p):
                pilotInterference[t] = np.sum(db2pow(gainOverNoisedB[master,:k ][pilotIndex[:k]==t]))

            # Find the pilot with least received power
            bestPilot = np.argmin(pilotInterference)
            pilotIndex[k] = bestPilot

        for l in range(L):      # Go through all APs
            # -------------------------- MODIFIED --------------------------------
            # UEpositions/APpositions are (K,1)/(L,1) arrays, so single-axis
            # indexing returns shape-(1,) arrays. NumPy 2.x refuses to
            # implicitly convert those to Python scalars inside
            # localScatteringR, raising TypeError. Explicit 2D indexing
            # keeps this a true scalar on both old and new NumPy.
            angletoUE_varphi = np.angle(UEpositions[k, 0]-APpositions[l, 0])
            angletoUE_theta = np.arcsin(distanceVertical/distances[l,k])
            # Generate the approximate spatial correlation matrix using the local scattering model by scaling
            # the normalized matrices with the channel gain
            # print(f'R: {R}')
            if l in closest_APs:
                R[:, :, l, k] = db2pow(gainOverNoisedB[l, k])*localScatteringR(N, angletoUE_varphi, ASD_varphi, antennaSpacing)
            # print(f'R for l {l}: {R}')

    # -------------------------- MODIFIED --------------------------------
    # pilotIndex is now returned (as ints) alongside the rest: it was
    # already computed above but silently discarded before. It is needed
    # downstream to build pilot-contamination edges in the conflict graph.
    pilotIndex = pilotIndex.astype(int)

    return gainOverNoisedB, powgain, active_APs, distances, pilotIndex # APpositions, UEpositions


def db2pow(dB):

    pow = 10**(dB/10)
    return pow

def localScatteringR(N, nominalAngle, ASD, antennaSpacing):

    firstColumn = np.zeros((N), dtype=complex)

    for column in range(N):
        distance = column

        firstColumn[column] = np.exp(1j * 2 * np.pi * antennaSpacing * np.sin(nominalAngle) * distance) * np.exp(
            (-(ASD ** 2) / 2) * (2 * np.pi * antennaSpacing * np.cos(nominalAngle) * distance) ** 2)

    R = spalg.toeplitz(firstColumn)

    return np.matrix(R).T
