-- =====================================================================
-- EcoSort AI: Smart Waste Segregation System (SDG 12)
-- Database Schema for MySQL
-- Database Name: ecosort_ai
-- =====================================================================

-- Step 1: Create Database
CREATE DATABASE IF NOT EXISTS `ecosort_ai`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE `ecosort_ai`;

-- Step 2: Drop Tables in reverse dependency order (if re-running schema)
DROP TABLE IF EXISTS `prediction_feedback`;
DROP TABLE IF EXISTS `predictions`;
DROP TABLE IF EXISTS `waste_categories`;
DROP TABLE IF EXISTS `users`;

-- =====================================================================
-- Table 1: users
-- Stores registered users and administrators
-- =====================================================================
CREATE TABLE `users` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `name` VARCHAR(100) NOT NULL,
    `email` VARCHAR(150) NOT NULL,
    `password_hash` VARCHAR(255) NOT NULL,
    `role` ENUM('user', 'admin') NOT NULL DEFAULT 'user',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY `uq_users_email` (`email`),
    INDEX `idx_users_role` (`role`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- Table 2: waste_categories
-- Master lookup table for the 4 core waste classification categories
-- =====================================================================
CREATE TABLE `waste_categories` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `category_name` VARCHAR(50) NOT NULL,
    `bin_name` VARCHAR(60) NOT NULL,
    `bin_color_code` VARCHAR(10) NOT NULL,
    `description` TEXT NOT NULL,
    `disposal_instructions` TEXT NOT NULL,
    `examples` TEXT NOT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY `uq_category_name` (`category_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- Table 3: predictions
-- Stores image classification records and prediction confidence
-- =====================================================================
CREATE TABLE `predictions` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `user_id` INT NULL,
    `image_filename` VARCHAR(255) NOT NULL,
    `predicted_category_id` INT NOT NULL,
    `confidence_score` DECIMAL(5, 2) NOT NULL,
    `model_version` VARCHAR(50) NOT NULL DEFAULT 'MobileNetV2-v1',
    `user_feedback` ENUM('unreviewed', 'correct', 'incorrect') NOT NULL DEFAULT 'unreviewed',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT `fk_predictions_user`
        FOREIGN KEY (`user_id`)
        REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    CONSTRAINT `fk_predictions_category`
        FOREIGN KEY (`predicted_category_id`)
        REFERENCES `waste_categories` (`id`)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
    INDEX `idx_predictions_user` (`user_id`),
    INDEX `idx_predictions_category` (`predicted_category_id`),
    INDEX `idx_predictions_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- Table 4: prediction_feedback
-- Captures human-in-the-loop corrections to monitor real model quality
-- =====================================================================
CREATE TABLE `prediction_feedback` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `prediction_id` INT NOT NULL,
    `user_id` INT NULL,
    `corrected_category_id` INT NULL,
    `feedback_type` ENUM('confirm', 'correction', 'other') NOT NULL DEFAULT 'correction',
    `user_notes` TEXT NULL,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT `fk_feedback_prediction`
        FOREIGN KEY (`prediction_id`)
        REFERENCES `predictions` (`id`)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT `fk_feedback_user`
        FOREIGN KEY (`user_id`)
        REFERENCES `users` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    CONSTRAINT `fk_feedback_corrected_category`
        FOREIGN KEY (`corrected_category_id`)
        REFERENCES `waste_categories` (`id`)
        ON DELETE SET NULL
        ON UPDATE CASCADE,
    INDEX `idx_feedback_prediction` (`prediction_id`),
    INDEX `idx_feedback_created` (`created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- =====================================================================
-- Initial Seed Data: 4 Core Waste Categories
-- Aligned with SDG 12: Responsible Consumption and Production
-- =====================================================================
INSERT INTO `waste_categories` (`id`, `category_name`, `bin_name`, `bin_color_code`, `description`, `disposal_instructions`, `examples`) VALUES
(
    1,
    'Wet Waste',
    'Green Bin (Organic / Compostable)',
    '#28a745',
    'Biodegradable organic waste that decomposes naturally through microbial action.',
    'Dispose directly into the green bin without plastic liners or polythene bags. Ideal for home composting or municipal biomethanation processing.',
    'Fruit and vegetable peels, leftover food, tea bags, coffee grounds, eggshells, garden leaves, flowers.'
),
(
    2,
    'Dry Waste',
    'Blue Bin (Non-Recyclable Dry)',
    '#007bff',
    'Non-biodegradable, non-recyclable solid waste that cannot be composted or mechanically recycled.',
    'Ensure items are reasonably dry before placing in the blue bin to prevent odor and pest accumulation. Keep segregated from wet organic items.',
    'Soiled paper wrappers, non-recyclable multi-layered packaging, chip packets, broken ceramics, rubber scraps, sweeping dust, hygiene waste.'
),
(
    3,
    'Recyclable Waste',
    'Yellow Bin (Clean Recyclables)',
    '#ffc107',
    'Clean, non-contaminated materials that can be remanufactured into new products through industrial recycling.',
    'Rinse containers clean of food residue and air-dry thoroughly. Flatten cardboard boxes and crush plastic bottles to optimize transport volume.',
    'Clear PET plastic bottles, HDPE shampoo containers, aluminum beverage cans, tin cans, cardboard boxes, clean newspapers, magazines, glass bottles.'
),
(
    4,
    'E-Waste',
    'Red Bin (Specialized E-Waste Bin)',
    '#dc3545',
    'Discarded electrical and electronic equipment containing toxic heavy metals (lead, cadmium, mercury) requiring controlled recovery.',
    'DO NOT discard with household garbage or landfill bins. Store safely in a dry location and hand over to authorized e-waste collection centers or recyclers.',
    'Mobile phones, chargers, USB cables, earphones, remote controls, printed circuit boards (PCBs), alkaline/lithium batteries, laptop accessories.'
);
