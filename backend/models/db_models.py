from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Date, BigInteger, ForeignKey
from sqlalchemy.orm import relationship
from core.database import Base

class Client(Base):
    __tablename__ = "clients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    postal_address = Column(Text, nullable=True)
    logo = Column(Text, nullable=True)
    tva_nbr = Column(String(100), nullable=True)

    sites = relationship("Site", back_populates="client")

    def __repr__(self):
        return f"<Client(id={self.id}, name='{self.name}')>"

class Site(Base):
    __tablename__ = "sites"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255))
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    address = Column(Text, nullable=True)
    project = Column(String(100), nullable=True)

    client = relationship("Client", back_populates="sites")
    devices = relationship("Device", back_populates="site")
    daily_consumptions = relationship("DailyConsumption", back_populates="site")
    alarms = relationship("Alarm", back_populates="site")

    def __repr__(self):
        return f"<Site(id={self.id}, name='{self.name}')>"

class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    serial_number = Column(String(100))
    project = Column(String(100), nullable=True)
    enable = Column(Integer, default=0)
    last_connection = Column(DateTime, nullable=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=True)

    site = relationship("Site", back_populates="devices")
    channels = relationship("Channel", back_populates="device")
    anomalies = relationship("Anomaly", back_populates="device")

    def __repr__(self):
        return f"<Device(id={self.id}, serial_number='{self.serial_number}')>"

class Channel(Base):
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True, index=True)
    channel = Column(Integer, nullable=True)
    zone_config = Column(String(255), nullable=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=True)

    device = relationship("Device", back_populates="channels")
    channels_voltix = relationship("ChannelVoltix", back_populates="channel")
    consumptions = relationship("Consumption", back_populates="channel_rel")

    def __repr__(self):
        return f"<Channel(id={self.id}, channel={self.channel})>"

class ChannelVoltix(Base):
    __tablename__ = "channelsvoltix"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(Integer, ForeignKey("channels.id"))
    freq = Column(Float, nullable=True)
    tot_active_energ = Column(Float, nullable=True)
    tot_active_pow = Column(Float, nullable=True)
    tot_apparent_pow = Column(Float, nullable=True)
    tot_pow_factor = Column(Float, nullable=True)
    tot_reactive_energ = Column(Float, nullable=True)
    tot_reactive_pow = Column(Float, nullable=True)
    measure_id = Column(Integer, nullable=True)

    channel = relationship("Channel", back_populates="channels_voltix")

    def __repr__(self):
        return f"<ChannelVoltix(id={self.id})>"

class Consumption(Base):
    __tablename__ = "consumptions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(BigInteger)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=True)
    consumption_diff = Column(Float, nullable=True)
    sn = Column(String(100), nullable=True)
    tot_active_energ = Column(Float, nullable=True)

    channel_rel = relationship("Channel", back_populates="consumptions")

    def __repr__(self):
        return f"<Consumption(id={self.id}, date={self.date})>"

class DailyConsumption(Base):
    __tablename__ = "dailyconsumptions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(BigInteger)
    local_date = Column(Date, nullable=True)
    total_consumption = Column(Float, nullable=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=True)

    site = relationship("Site", back_populates="daily_consumptions")

    def __repr__(self):
        return f"<DailyConsumption(id={self.id}, site_id={self.site_id})>"

class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)
    anomaly_name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    severity = Column(String(50), nullable=True)
    # Anomaly type: "Spike" or "Drop" based on reconstruction error direction
    anomaly_type = Column(String(20), nullable=True)
    start_date = Column(BigInteger, nullable=True)
    end_date = Column(BigInteger, nullable=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=True)
    project = Column(String(100), nullable=True)

    device = relationship("Device", back_populates="anomalies")

    def __repr__(self):
        return f"<Anomaly(id={self.id}, severity='{self.severity}')>"

class Alarm(Base):
    __tablename__ = "alarms"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=True)
    resolved = Column(Integer, default=0)
    resolved_date = Column(DateTime, nullable=True)
    threshold_id = Column(Integer, nullable=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=True)

    site = relationship("Site", back_populates="alarms")

    def __repr__(self):
        return f"<Alarm(id={self.id}, resolved={self.resolved})>"

class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("sites.id"), nullable=True)
    date = Column(BigInteger)
    predicted_value = Column(Float)
    model_used = Column(String(50))
    domain = Column(String(20))

    site = relationship("Site")

    def __repr__(self):
        return f"<Forecast(id={self.id}, domain='{self.domain}', date={self.date})>"

class Report(Base):
    __tablename__ = "reports"

    id           = Column(Integer, primary_key=True, index=True)
    domain       = Column(String(20))    # "energy" | "water" | "combined"
    source       = Column(String(20))    # "manual" | "weekly_cron"
    report_text  = Column(Text)
    generated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f"<Report(id={self.id}, domain='{self.domain}', source='{self.source}')>"
