import pandas as pd
import numpy as np
from datetime import datetime


class RescueModule:
    def __init__(self):
        self.rescue_plans = {
            'high_risk': {
                'immediate_actions': [
                    "Evacuate to higher ground immediately",
                    "Switch off electrical mains",
                    "Take emergency kit and important documents",
                    "Avoid walking or driving through flood waters"
                ],
                'authorities': [
                    "National Disaster Response Force: 1070",
                    "State Disaster Control Room",
                    "Local Police: 100",
                    "Ambulance: 108"
                ],
                'shelters': [
                    "Designated flood shelters in area",
                    "Multi-story concrete buildings",
                    "Schools and community centers on high ground"
                ]
            },
            'medium_risk': {
                'preparations': [
                    "Prepare emergency evacuation bag",
                    "Monitor weather updates regularly",
                    "Secure important items to higher levels",
                    "Keep vehicles fueled and ready"
                ],
                'contacts': [
                    "Local administration office",
                    "Meteorological department updates",
                    "Neighborhood watch groups"
                ],
                'checks': [
                    "Check drainage systems around property",
                    "Ensure emergency supplies are stocked",
                    "Identify safe routes to higher ground"
                ]
            },
            'low_risk': {
                'monitoring': [
                    "Continue normal activities with caution",
                    "Stay informed about weather forecasts",
                    "Keep emergency contacts handy"
                ],
                'precautions': [
                    "Avoid venturing into low-lying areas",
                    "Monitor water levels in nearby rivers",
                    "Have basic emergency supplies ready"
                ]
            }
        }

        self.resource_mapping = {
            'medical': {
                'priority': 1,
                'contacts': ["Hospital", "Medical camps", "First aid centers"],
                'requirements': ["Medicines", "First aid kits", "Medical personnel"]
            },
            'evacuation': {
                'priority': 2,
                'contacts': ["Transport department", "Bus operators", "Boat services"],
                'requirements': ["Vehicles", "Boats", "Helicopters"]
            },
            'shelter': {
                'priority': 3,
                'contacts': ["Municipal corporation", "NGOs", "Community centers"],
                'requirements': ["Temporary shelters", "Food", "Water", "Blankets"]
            },
            'communication': {
                'priority': 4,
                'contacts': ["Telecom companies", "Radio stations", "Local authorities"],
                'requirements': ["Satellite phones", "Radio equipment", "Charging stations"]
            }
        }

    def assess_risk_level(self, prediction_probability, rainfall, water_level, population_density):
        risk_score = 0

        if prediction_probability >= 0.8:
            risk_score += 3
        elif prediction_probability >= 0.6:
            risk_score += 2
        elif prediction_probability >= 0.4:
            risk_score += 1

        if rainfall > 200:
            risk_score += 2
        elif rainfall > 100:
            risk_score += 1

        if water_level > 8:
            risk_score += 3
        elif water_level > 5:
            risk_score += 2
        elif water_level > 3:
            risk_score += 1

        if population_density > 5000:
            risk_score += 2
        elif population_density > 2000:
            risk_score += 1

        if risk_score >= 8:
            return 'high_risk'
        elif risk_score >= 5:
            return 'medium_risk'
        else:
            return 'low_risk'

    def generate_rescue_plan(self, prediction_data, location_info=None):
        probability = prediction_data.get('probability', 0)
        rainfall = prediction_data.get('rainfall', 0)
        water_level = prediction_data.get('water_level', 0)
        population_density = prediction_data.get('population_density', 0)

        risk_level = self.assess_risk_level(probability, rainfall, water_level, population_density)
        rescue_plan = self.rescue_plans[risk_level]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        plan = {
            'timestamp': timestamp,
            'risk_level': risk_level,
            'risk_score': self.calculate_risk_score(probability, rainfall, water_level, population_density),
            'immediate_actions': rescue_plan.get('immediate_actions', []) if risk_level == 'high_risk' else [],
            'preparations': rescue_plan.get('preparations', []) if risk_level == 'medium_risk' else [],
            'monitoring_actions': rescue_plan.get('monitoring', []) if risk_level == 'low_risk' else [],
            'authorities_to_contact': rescue_plan.get('authorities', []),
            'resource_requirements': self.identify_resource_needs(risk_level, location_info),
            'evacuation_routes': self.suggest_evacuation_routes(location_info),
            'safety_checklist': self.generate_safety_checklist(risk_level)
        }

        return plan

    def calculate_risk_score(self, probability, rainfall, water_level, population_density):
        base_score = probability * 100
        environmental_factor = (rainfall / 300 * 25) + (water_level / 10 * 25)
        social_factor = min(population_density / 10000 * 50, 50)

        return min(base_score + environmental_factor + social_factor, 100)

    def identify_resource_needs(self, risk_level, location_info):
        resources = {}

        if risk_level == 'high_risk':
            resources = {
                'critical': ['emergency_services', 'medical_aid', 'evacuation_transport'],
                'essential': ['food_supplies', 'clean_water', 'shelter'],
                'support': ['communication_devices', 'power_supplies', 'rescue_equipment']
            }
        elif risk_level == 'medium_risk':
            resources = {
                'preparedness': ['emergency_kits', 'communication_gear', 'medical_supplies'],
                'monitoring': ['weather_stations', 'water_level_sensors', 'communication_networks']
            }
        else:
            resources = {
                'basic': ['emergency_contacts', 'basic_supplies', 'information_sources']
            }

        if location_info and location_info.get('urban_area'):
            resources['infrastructure'] = ['multi_story_buildings', 'hospitals', 'communication_towers']
        elif location_info and location_info.get('rural_area'):
            resources['infrastructure'] = ['high_ground', 'community_centers', 'access_roads']

        return resources

    def suggest_evacuation_routes(self, location_info):
        routes = []

        if not location_info:
            routes.append("Move to highest available ground in area")
            routes.append("Follow designated flood evacuation signs")
            return routes

        if location_info.get('urban_area'):
            routes.append("Use main roads to designated shelters")
            routes.append("Avoid underground passages and subways")
            routes.append("Head to multi-story concrete buildings")

        if location_info.get('coastal_area'):
            routes.append("Move inland away from coastline")
            routes.append("Use elevated roads and bridges")
            routes.append("Follow tsunami evacuation routes if applicable")

        if location_info.get('riverine_area'):
            routes.append("Move perpendicular to river flow direction")
            routes.append("Seek high ground away from river banks")
            routes.append("Avoid low-lying bridges and crossings")

        routes.append("Always follow instructions from local authorities")

        return routes

    def generate_safety_checklist(self, risk_level):
        checklist = {}

        if risk_level == 'high_risk':
            checklist = {
                'before_evacuation': [
                    "Turn off electricity, gas, and water",
                    "Secure important documents in waterproof bags",
                    "Take emergency kit and medications",
                    "Inform relatives about evacuation plan"
                ],
                'during_evacuation': [
                    "Do not walk through moving water",
                    "Avoid driving through flooded areas",
                    "Stay on firm ground",
                    "Follow official evacuation routes"
                ],
                'after_evacuation': [
                    "Wait for official all-clear before returning",
                    "Check for structural damage to property",
                    "Boil water before drinking",
                    "Document damages for insurance"
                ]
            }
        elif risk_level == 'medium_risk':
            checklist = {
                'preparation': [
                    "Prepare emergency evacuation bag",
                    "Identify safe meeting points",
                    "Keep vehicles fueled",
                    "Charge all communication devices"
                ],
                'monitoring': [
                    "Monitor official weather updates",
                    "Watch water levels in local water bodies",
                    "Stay connected with neighbors",
                    "Keep emergency numbers accessible"
                ]
            }
        else:
            checklist = {
                'awareness': [
                    "Stay informed about weather conditions",
                    "Know your local evacuation routes",
                    "Keep basic emergency supplies ready",
                    "Save important contact numbers"
                ]
            }

        return checklist

    def get_emergency_contacts(self, state=None, district=None):
        contacts = {
            'national': {
                'NDRF': '1070',
                'Disaster Management': '1078',
                'Police': '100',
                'Ambulance': '108',
                'Fire': '101'
            }
        }

        state_contacts = {
            'maharashtra': {'Flood Control Room': '022-22027990'},
            'kerala': {'Disaster Management': '0471-2364424'},
            'tamil nadu': {'Emergency Operations': '044-25678506'},
            'west bengal': {'Disaster Management': '033-22143526'},
            'karnataka': {'Control Room': '080-22253707'}
        }

        if state and state.lower() in state_contacts:
            contacts['state'] = state_contacts[state.lower()]

        return contacts

    def generate_alert_message(self, risk_level, probability, location_name=""):
        if risk_level == 'high_risk':
            return f"🚨 FLOOD ALERT for {location_name}! Immediate evacuation recommended. Probability: {probability:.1%}"
        elif risk_level == 'medium_risk':
            return f"⚠️ FLOOD WARNING for {location_name}. Prepare for possible evacuation. Probability: {probability:.1%}"
        else:
            return f"ℹ️ FLOOD ADVISORY for {location_name}. Stay alert and monitor conditions. Probability: {probability:.1%}"


def create_rescue_assistance(prediction_result, sensor_data=None, location_data=None):
    rescue_system = RescueModule()

    prediction_data = {
        'probability': prediction_result.get('probability', 0),
        'rainfall': sensor_data.get('rainfall', 0) if sensor_data else 0,
        'water_level': sensor_data.get('water_level', 0) if sensor_data else 0,
        'population_density': location_data.get('population_density', 0) if location_data else 0
    }

    rescue_plan = rescue_system.generate_rescue_plan(prediction_data, location_data)
    alert_message = rescue_system.generate_alert_message(
        rescue_plan['risk_level'],
        prediction_data['probability'],
        location_data.get('name', 'the area') if location_data else 'the area'
    )

    return {
        'alert': alert_message,
        'rescue_plan': rescue_plan,
        'emergency_contacts': rescue_system.get_emergency_contacts(
            location_data.get('state') if location_data else None
        ),
        'timestamp': datetime.now().isoformat()
    }