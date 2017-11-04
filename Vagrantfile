Vagrant.configure("2") do |config|

  config.vm.define "touchnet" do | tn |
    tn.vm.box = "ubuntu/xenial64"
    tn.vm.hostname = "touchnet"
    tn.vm.synced_folder ".", "/vagrant"

    tn.vm.provision "shell", path: "provision/setup.sh"
    tn.vm.provider "virtualbox" do | pv |
      pv.memory = "1024"
      pv.cpus = 2
      pv.gui = true
    end
  end
end

