import heapq
from Distribution import Distribution
from HydroRelease import HydroRelease
from SolarGeneration import SolarGeneration
from Notification import Notification
from DayInfo import DayInfo
# Initialize our variables
# We can make this as system day variables
# Time may be bit confusing since ig we can assume this is when we are planning to make an informed decision.
# [Day1, Day2, Day3] -> We can maybe use a priority queue to determine how much to release based on higher demand days?
# This will allocate water to days that we expect the most hydro releases. In general since we can assign a priorty I
# I think this may be the way to do it.
# Or we can do day to day meeting requirements unless it doesnt.
'''
Day 1 Info
energyDemand = 900
time = 12

panelEfficency = .9

inflow = 1000
waterReleased = 0
hydroEnergy = 10

hydroWeight = .5
solarWeight = .5
'''
hydroStored = 1000
panelAmount = 5
day1 = DayInfo(900,12,.9,panelAmount,1000,hydroStored,0,10,.5,.5)
day2 = DayInfo(1000,9,.9,panelAmount,50,hydroStored,0,10,.5,.5)
weekInfomation = [day1,day2]
heapq.heapify(weekInfomation)
# Min difference between actual and demand
# Water rights are annual and some per month
# General need however, 
# Price varys hourly
# Stabilization Attempts
dayRelease = []
while weekInfomation:
    attempts = 0
    satisfies = False
    while attempts < 3 and not satisfies:
        # Get Day Information
        day = heapq.heappop(weekInfomation)
        # Generate Distribution
        distribution = Distribution(day.energyDemand, day.hydroWeight, day.solarWeight)
        hydroDemand, solarDemand = distribution.calculateRequired()

        # Determine System Generation Predictions
        hydroRelease = HydroRelease(day.inflow, day.time, hydroStored)
        waterReleased, hydroReleasePrediction = hydroRelease.releasePrediction()
        solarRelease = SolarGeneration(panelAmount, day.panelEfficency, day.time)
        solarGenerationPrediction = solarRelease.generatePrediction()
        print(hydroReleasePrediction, solarGenerationPrediction)

        # Evaluate
        evaluator = Notification(day.energyDemand, hydroReleasePrediction, solarGenerationPrediction)
        satisfies, hydroWeight, solarWeight = evaluator.evaluate(day.hydroWeight, day.solarWeight)
        if satisfies:
            dayRelease.append((day, waterReleased))
        attempts += 1
    if not satisfies:
        heapq.heappush(weekInfomation, day)
        break


# Release System
if not weekInfomation:
    for day, waterReleased in dayRelease:
        print("Day: ", day.time, "Water Released: ", waterReleased)
    hydroStored -= waterReleased
    print("System Met Required Demand")
else:
    print("System Cannot Meet Requirements")
