#include <Servo.h>

Servo myservo ; 
int angle = 90;  // Initial angle

void setup() {
  myservo.attach(5);  // Attach the servo to pin 5
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    

    if (command == 'a') {  // Command for apple
      myservo.writeMicroseconds(2000); // Set servo angle for red
    } else if (command == 'b') {  // Command for banana
      myservo.writeMicroseconds(1000); // Set servo angle for red
    }

  }
}

